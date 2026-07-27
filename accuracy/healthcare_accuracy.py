import os
import pymysql
import logging
import json
import re
import threading
import datetime as dt
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence, Optional, Dict
from dataclasses import dataclass
from langfuse import Langfuse
try:
    import httpx as _httpx
except ImportError:
    _httpx = None

# Attempt importing clickhouse_store directly, falling back to parent folder injection if needed
try:
    from clickhouse_store import (
        get_environment,
        insert_healthcare_accuracy_rows,
        HEALTHCARE_EOB_ACCURACY_TABLE,
        HEALTHCARE_SUPERBILL_ACCURACY_TABLE
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from clickhouse_store import (
        get_environment,
        insert_healthcare_accuracy_rows,
        HEALTHCARE_EOB_ACCURACY_TABLE,
        HEALTHCARE_SUPERBILL_ACCURACY_TABLE
    )

# ---------------------------------------------------------------------------
# Data Classes and Global/Thread-Local Configuration
# ---------------------------------------------------------------------------

@dataclass
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

@dataclass
class CompareStats:
    matches: int = 0
    mismatches: int = 0

_EOB_ACTIVE_STATS: Optional[CompareStats] = None
_SB_ACTIVE_STATS = threading.local()

IGNORED_MISMATCH_SNIPPETS = (
    ".Provider role=BillingProvider count mismatch.",
    ".Provider role=ServicingProvider count mismatch.",
    ".Provider role=ServiceFacilityProvider count mismatch.",
    ".Provider count mismatch.",
    "Allocation.File_Status mismatch.",
    "Allocation.Completed_Date mismatch.",
    ".Payer count mismatch.",
)

# ---------------------------------------------------------------------------
# Helper Utility Functions for Audit Comparisons
# ---------------------------------------------------------------------------

def _parse_jdbc(jdbc_url: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    match = re.match(r"^jdbc:mysql://([^/:?#]+)(?::(\d+))?/([^?]+)", jdbc_url.strip(), flags=re.IGNORECASE)
    if not match:
        return None, None, None
    host = match.group(1)
    port = int(match.group(2)) if match.group(2) else None
    database = match.group(3)
    return host, port, database

def resolve_db_config() -> DbConfig:
    host = os.getenv("HEALTHCARE_AI_DB_SERVER")
    port_raw = os.getenv("HEALTHCARE_AI_DB_PORT")
    user = os.getenv("HEALTHCARE_AI_DB_USERID")
    password = os.getenv("HEALTHCARE_AI_DB_PASSWORD")
    database = os.getenv("HEALTHCARE_AI_DB_DATABASE")

    jdbc = os.getenv("HEALTHCARE_AI_DB_JDBC_URL")
    if jdbc:
        jdbc_host, jdbc_port, jdbc_db = _parse_jdbc(jdbc)
        host = host or jdbc_host
        if port_raw is None and jdbc_port is not None:
            port_raw = str(jdbc_port)
        database = database or jdbc_db

    if not host or not user or not password or not database:
        raise ValueError(
            "Missing DB config. Set HEALTHCARE_AI_DB_SERVER/USERID/PASSWORD/DATABASE "
            "or HEALTHCARE_AI_DB_JDBC_URL plus missing pieces."
        )

    port = int(port_raw) if port_raw else 3306
    return DbConfig(host=host, port=port, user=user, password=password, database=database)

def q_ident(name: str) -> str:
    return f"`{name}`"

def resolve_table_name(conn: pymysql.connections.Connection, candidates: Sequence[str]) -> str:
    sql = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = %s
    """
    db_name = conn.db.decode() if isinstance(conn.db, bytes) else conn.db
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(sql, (db_name,))
        rows = cur.fetchall()

    existing = {row["TABLE_NAME"].lower(): row["TABLE_NAME"] for row in rows}
    for candidate in candidates:
        hit = existing.get(candidate.lower())
        if hit:
            return hit
    raise RuntimeError(f"None of table candidates exist: {candidates}")

def fetch_rows_by_fk(
    conn: pymysql.connections.Connection,
    table: str,
    fk_name: str,
    fk_value: Any,
    order_by: str = "Id",
) -> list[dict[str, Any]]:
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            f"SELECT * FROM {q_ident(table)} WHERE {q_ident(fk_name)}=%s ORDER BY {q_ident(order_by)}",
            (fk_value,),
        )
        return cur.fetchall()

def fetch_row_by_id(conn: pymysql.connections.Connection, table: str, row_id: Any) -> Optional[dict[str, Any]]:
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(f"SELECT * FROM {q_ident(table)} WHERE Id=%s", (row_id,))
        return cur.fetchone()

def parse_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None

def parse_int(value: Optional[str]) -> Optional[int]:
    stripped = parse_str(value)
    if stripped is None:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None

def parse_money(value: Optional[str]) -> Optional[Decimal]:
    stripped = parse_str(value)
    if stripped is None:
        return None
    cleaned = stripped.replace("$", "").replace(",", "").strip()
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None

def parse_date(value: Optional[str]) -> Optional[dt.date]:
    stripped = parse_str(value)
    if stripped is None:
        return None
    for fmt in ["%m-%d-%Y", "%Y-%m-%d", "%m%d%Y", "%Y%m%d", "%m/%d/%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None

def ocr_value(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    value = node.get("value")
    if value is None:
        value = node.get("Value")
    return str(value) if value is not None else None

def ocr_value_any(container: Optional[dict[str, Any]], *keys: str) -> Optional[str]:
    if not isinstance(container, dict):
        return None
    for key in keys:
        if key in container:
            return ocr_value(container.get(key))
    return None

def map_file_status(value: Optional[str]) -> str:
    if value == "Pending":
        return "Pending"
    if value == "Completed":
        return "Completed"
    if value == "Failed":
        return "Failed"
    if value in ("Manually Created", "ManuallyCreated"):
        return "ManuallyCreated"
    if value in ("Partially Completed", "PartiallyCompleted"):
        return "PartiallyCompleted"
    return "Pending"

def to_date_db(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return parse_date(value)
    return None

def to_decimal_db(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None

def normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    FILE_STATUS_ENUM = {
        0: "Pending",
        1: "Failed",
        2: "Completed",
        3: "ManuallyCreated",
        4: "PartiallyCompleted",
    }
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in FILE_STATUS_ENUM.values():
            return stripped
        if stripped.isdigit():
            return FILE_STATUS_ENUM.get(int(stripped), stripped)
        return stripped
    if isinstance(value, int):
        return FILE_STATUS_ENUM.get(value, str(value))
    return str(value)

def normalize_enum(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)

def normalize_for_compare(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped == "" else stripped
    return value

def normalize_compare_str(value: str) -> str:
    return " ".join(value.strip().split()).casefold()

def alnum_only(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())

def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)

def is_diagnosis_path(path: str) -> bool:
    return (
        ".Diagnosis." in path
        or path.endswith(".Primary_DX")
        or ".Secondary_DX" in path
        or re.search(r"\.DX_[1-4]$", path) is not None
    )

def is_phone_path(path: str) -> bool:
    return "Phone_No" in path

def is_zip_path(path: str) -> bool:
    return path.endswith(".Patient_Zip") or path.endswith(".Payer_Zip") or path.endswith(".Provider_Zip")

def value_str(value: Any) -> str:
    return "<null>" if value is None else str(value)

def should_ignore_mismatch(message: str) -> bool:
    return any(snippet in message for snippet in IGNORED_MISMATCH_SNIPPETS)

# ---------------------------------------------------------------------------
# Date Extraction and Formatting Helpers
# ---------------------------------------------------------------------------

def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None

def _extract_date_string(value: Any) -> Optional[str]:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.date().isoformat()
    if isinstance(value, str):
        raw = value.strip()
        if len(raw) >= 10:
            return raw[:10]
    return None

def _to_iso_datetime_string(value: Any) -> Optional[str]:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.isoformat(sep=" ")
    if value is None:
        return None
    return str(value)

# ---------------------------------------------------------------------------
# EOB Audit Core Comparison Functions
# ---------------------------------------------------------------------------

def eob_add_mismatch(message: str, mismatches: list[str]) -> None:
    global _EOB_ACTIVE_STATS
    mismatches.append(message)
    if _EOB_ACTIVE_STATS is not None:
        _EOB_ACTIVE_STATS.mismatches += 1

def eob_eq(path: str, actual: Any, expected: Any, mismatches: list[str]) -> None:
    global _EOB_ACTIVE_STATS
    if normalize_for_compare(actual) == normalize_for_compare(expected):
        if _EOB_ACTIVE_STATS is not None:
            _EOB_ACTIVE_STATS.matches += 1
        return
    eob_add_mismatch(f"{path} mismatch. expected={value_str(expected)}, actual={value_str(actual)}", mismatches)

def audit_eob_allocation(
    allocation_row: dict[str, Any],
    conn: pymysql.connections.Connection,
    tables: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    raw_json = allocation_row.get("rawJson") or allocation_row.get("RawJson")
    if not raw_json:
        eob_add_mismatch("rawJson missing on allocation row.", mismatches)
        return mismatches

    try:
        root = json.loads(raw_json)
    except json.JSONDecodeError as ex:
        eob_add_mismatch(f"rawJson parse error: {ex}", mismatches)
        return mismatches

    allocation = root.get("Allocation")
    if not isinstance(allocation, dict):
        eob_add_mismatch("rawJson does not contain Allocation object.", mismatches)
        return mismatches

    eob_eq("Allocation.File_name", allocation_row.get("File_name"), allocation.get("File_name"), mismatches)
    eob_eq("Allocation.File_url", allocation_row.get("File_url"), allocation.get("File_url"), mismatches)
    eob_eq("Allocation.Client", allocation_row.get("Client"), allocation.get("Client"), mismatches)
    eob_eq("Allocation.Account", allocation_row.get("Account"), allocation.get("Account"), mismatches)
    eob_eq("Allocation.Payer", allocation_row.get("Payer"), parse_str(ocr_value(allocation.get("Payer"))), mismatches)
    eob_eq("Allocation.Total_No_Of_Clm_On_File", allocation_row.get("Total_No_Of_Clm_On_File"), allocation.get("Total_No_Of_Clm_On_File"), mismatches)
    eob_eq("Allocation.Total_Paid_Amt_On_File", to_decimal_db(allocation_row.get("Total_Paid_Amt_On_File")), parse_money(allocation.get("Total_Paid_Amt_On_File")), mismatches)
    eob_eq("Allocation.Check_Date", to_date_db(allocation_row.get("Check_Date")), parse_date(allocation.get("Check_Date")), mismatches)
    eob_eq("Allocation.Total_Denied_Claims", allocation_row.get("Total_Denied_Claims"), allocation.get("Total_Denied_Claims"), mismatches)
    eob_eq("Allocation.Total_Denied_Lines", allocation_row.get("Total_Denied_Lines"), allocation.get("Total_Denied_Lines"), mismatches)
    eob_eq("Allocation.Total_Posted_Claims", allocation_row.get("Total_Posted_Claims"), parse_int(allocation.get("Total_Posted_Claims")), mismatches)
    eob_eq("Allocation.Download_Date", to_date_db(allocation_row.get("Download_Date")), parse_date(allocation.get("Download_Date")), mismatches)
    eob_eq("Allocation.Completed_Date", to_date_db(allocation_row.get("Completed_Date")), parse_date(allocation.get("Completed_Date")), mismatches)
    eob_eq("Allocation.Not_Completed_Reasons", allocation_row.get("Not_Completed_Reasons"), allocation.get("Not_Completed_Reasons"), mismatches)
    eob_eq("Allocation.File_Status", normalize_status(allocation_row.get("File_Status")), map_file_status(allocation.get("File_Status")), mismatches)

    expected_claims = []
    for claim_info in allocation.get("Claims_Info") or []:
        if isinstance(claim_info, dict) and isinstance(claim_info.get("Claim"), dict):
            expected_claims.append(claim_info["Claim"])

    db_claims = fetch_rows_by_fk(conn, tables["claim"], "AllocationId", allocation_row["Id"], order_by="Id")
    if len(db_claims) != len(expected_claims):
        eob_add_mismatch(f"Claims count mismatch. expected={len(expected_claims)}, actual={len(db_claims)}", mismatches)

    for idx, (db_claim, claim_obj) in enumerate(zip(db_claims, expected_claims)):
        p = f"Claim[{idx}]"
        eob_eq(f"{p}.Claim_Number", db_claim.get("Claim_Number"), parse_str(ocr_value(claim_obj.get("Claim_Number"))), mismatches)
        eob_eq(f"{p}.Claim_Total_Charge_Amt", to_decimal_db(db_claim.get("Claim_Total_Charge_Amt")), parse_money(ocr_value(claim_obj.get("Claim_Total_Charge_Amt"))), mismatches)
        eob_eq(f"{p}.Claim_Paid_Amt", to_decimal_db(db_claim.get("Claim_Paid_Amt")), parse_money(ocr_value(claim_obj.get("Claim_Paid_Amt"))), mismatches)
        eob_eq(f"{p}.Claim_Status_Code", db_claim.get("Claim_Status_Code"), parse_str(ocr_value(claim_obj.get("Claim_Status_Code"))), mismatches)
        eob_eq(f"{p}.Claim_Status_Reason", db_claim.get("Claim_Status_Reason"), parse_str(ocr_value(claim_obj.get("Claim_Status_Reason"))), mismatches)
        eob_eq(f"{p}.Claim_Facility_Type", db_claim.get("Claim_Facility_Type"), parse_str(ocr_value(claim_obj.get("Claim_Facility_Type"))), mismatches)
        eob_eq(f"{p}.Claim_Frequency", db_claim.get("Claim_Frequency"), parse_str(ocr_value(claim_obj.get("Claim_Frequency"))), mismatches)
        eob_eq(f"{p}.Claim_Date_of_Service_From", to_date_db(db_claim.get("Claim_Date_of_Service_From")), parse_date(ocr_value(claim_obj.get("Claim_Date_of_Service_From"))), mismatches)
        eob_eq(f"{p}.Claim_Date_of_Service_To", to_date_db(db_claim.get("Claim_Date_of_Service_To")), parse_date(ocr_value(claim_obj.get("Claim_Date_of_Service_To"))), mismatches)
        eob_eq(f"{p}.Claim_Recieved_Date", to_date_db(db_claim.get("Claim_Recieved_Date")), parse_date(ocr_value(claim_obj.get("Claim_Recieved_Date"))), mismatches)
        eob_eq(f"{p}.Patient_Responsibility_Amt", to_decimal_db(db_claim.get("Patient_Responsibility_Amt")), parse_money(ocr_value(claim_obj.get("Patient_Responsibility_Amt"))), mismatches)
        eob_eq(f"{p}.CLIA_Number", db_claim.get("CLIA_Number"), parse_str(ocr_value(claim_obj.get("CLIA_Number"))), mismatches)
        eob_eq(f"{p}.Claim_Invoice_Number", db_claim.get("Claim_Invoice_Number"), parse_str(ocr_value(claim_obj.get("Claim_Invoice_Number"))), mismatches)
        eob_eq(f"{p}.Admission_Date", to_date_db(db_claim.get("Admission_Date")), parse_date(ocr_value(claim_obj.get("Admission_Date"))), mismatches)
        eob_eq(f"{p}.Patient_Last_Seen_Date", to_date_db(db_claim.get("Patient_Last_Seen_Date")), parse_date(ocr_value(claim_obj.get("Patient_Last_Seen_Date"))), mismatches)

        expected_payer = claim_obj.get("Payer")
        db_payer = fetch_row_by_id(conn, tables["payer"], db_claim.get("PayerId")) if db_claim.get("PayerId") else None
        if isinstance(expected_payer, dict):
            if db_payer is None:
                eob_add_mismatch(f"{p}.Payer missing in DB (expected present).", mismatches)
            else:
                pp = f"{p}.Payer"
                eob_eq(f"{pp}.Payer_Name", db_payer.get("Payer_Name"), parse_str(ocr_value(expected_payer.get("Payer_Name"))), mismatches)
                eob_eq(f"{pp}.Payer_Id", db_payer.get("Payer_Id"), parse_str(ocr_value(expected_payer.get("Payer_Id"))), mismatches)
                eob_eq(f"{pp}.Payer_Addr1", db_payer.get("Payer_Addr1"), parse_str(ocr_value(expected_payer.get("Payer_Addr1"))), mismatches)
                eob_eq(f"{pp}.Payer_Addr2", db_payer.get("Payer_Addr2"), parse_str(ocr_value(expected_payer.get("Payer_Addr2"))), mismatches)
                eob_eq(f"{pp}.Payer_City", db_payer.get("Payer_City"), parse_str(ocr_value(expected_payer.get("Payer_City"))), mismatches)
                eob_eq(f"{pp}.Payer_State", db_payer.get("Payer_State"), parse_str(ocr_value(expected_payer.get("Payer_State"))), mismatches)
                eob_eq(f"{pp}.Payer_Zip", db_payer.get("Payer_Zip"), parse_str(ocr_value(expected_payer.get("Payer_Zip"))), mismatches)
        elif db_payer is not None:
            eob_add_mismatch(f"{p}.Payer present in DB but missing in rawJson.", mismatches)

        expected_payee = claim_obj.get("Payee")
        db_payee = fetch_row_by_id(conn, tables["payee"], db_claim.get("PayeeId")) if db_claim.get("PayeeId") else None
        if isinstance(expected_payee, dict):
            if db_payee is None:
                eob_add_mismatch(f"{p}.Payee missing in DB (expected present).", mismatches)
            else:
                pe = f"{p}.Payee"
                eob_eq(f"{pe}.Payee_Name", db_payee.get("Payee_Name"), parse_str(ocr_value(expected_payee.get("Payee_Name"))), mismatches)
                eob_eq(f"{pe}.Payee_NPI", db_payee.get("Payee_NPI"), parse_str(ocr_value(expected_payee.get("Payee_NPI"))), mismatches)
                eob_eq(f"{pe}.Payee_TaxID", db_payee.get("Payee_TaxID"), parse_str(ocr_value(expected_payee.get("Payee_TaxID"))), mismatches)
                eob_eq(f"{pe}.Payee_Addr1", db_payee.get("Payee_Addr1"), parse_str(ocr_value(expected_payee.get("Payee_Addr1"))), mismatches)
                eob_eq(f"{pe}.Payee_Addr2", db_payee.get("Payee_Addr2"), parse_str(ocr_value(expected_payee.get("Payee_Addr2"))), mismatches)
                eob_eq(f"{pe}.Payee_City", db_payee.get("Payee_City"), parse_str(ocr_value(expected_payee.get("Payee_City"))), mismatches)
                eob_eq(f"{pe}.Payee_State", db_payee.get("Payee_State"), parse_str(ocr_value(expected_payee.get("Payee_State"))), mismatches)
                eob_eq(f"{pe}.Payee_Zip", db_payee.get("Payee_Zip"), parse_str(ocr_value(expected_payee.get("Payee_Zip"))), mismatches)
                eob_eq(f"{pe}.Rendering_Provider_Name", db_payee.get("Rendering_Provider_Name"), parse_str(ocr_value(expected_payee.get("Rendering_Provider_Name"))), mismatches)
                eob_eq(f"{pe}.Rendering_Provider_NPI", db_payee.get("Rendering_Provider_NPI"), parse_str(ocr_value(expected_payee.get("Rendering_Provider_NPI"))), mismatches)
                eob_eq(f"{pe}.Check_EFT_Number", db_payee.get("Check_EFT_Number"), parse_str(ocr_value(expected_payee.get("Check_EFT_Number"))), mismatches)
                eob_eq(f"{pe}.Payment_Amt", to_decimal_db(db_payee.get("Payment_Amt")), parse_money(ocr_value(expected_payee.get("Payment_Amt"))), mismatches)
                eob_eq(f"{pe}.Check_Date", to_date_db(db_payee.get("Check_Date")), parse_date(ocr_value(expected_payee.get("Check_Date"))), mismatches)
        elif db_payee is not None:
            eob_add_mismatch(f"{p}.Payee present in DB but missing in rawJson.", mismatches)

        expected_patient = claim_obj.get("Patient")
        db_patient = fetch_row_by_id(conn, tables["patient"], db_claim.get("PatientId")) if db_claim.get("PatientId") else None
        if isinstance(expected_patient, dict):
            if db_patient is None:
                eob_add_mismatch(f"{p}.Patient missing in DB (expected present).", mismatches)
            else:
                pt = f"{p}.Patient"
                eob_eq(f"{pt}.Patient_FN", db_patient.get("Patient_FN"), parse_str(ocr_value(expected_patient.get("Patient_FN"))), mismatches)
                eob_eq(f"{pt}.Patient_LN", db_patient.get("Patient_LN"), parse_str(ocr_value(expected_patient.get("Patient_LN"))), mismatches)
                eob_eq(f"{pt}.Patient_MI", db_patient.get("Patient_MI"), parse_str(ocr_value(expected_patient.get("Patient_MI"))), mismatches)
                eob_eq(f"{pt}.Patient_Id", db_patient.get("Patient_Id"), parse_str(ocr_value(expected_patient.get("Patient_Id"))), mismatches)
                eob_eq(f"{pt}.Patient_Control_Number", db_patient.get("Patient_Control_Number"), parse_str(ocr_value(expected_patient.get("Patient_Control_Number"))), mismatches)
                eob_eq(f"{pt}.Patient_Group", db_patient.get("Patient_Group"), parse_str(ocr_value(expected_patient.get("Patient_Group"))), mismatches)
                eob_eq(f"{pt}.Patient_Addr1", db_patient.get("Patient_Addr1"), parse_str(ocr_value(expected_patient.get("Patient_Addr1"))), mismatches)
                eob_eq(f"{pt}.Patient_Addr2", db_patient.get("Patient_Addr2"), parse_str(ocr_value(expected_patient.get("Patient_Addr2"))), mismatches)
                eob_eq(f"{pt}.Patient_City", db_patient.get("Patient_City"), parse_str(ocr_value(expected_patient.get("Patient_City"))), mismatches)
                eob_eq(f"{pt}.Patient_State", db_patient.get("Patient_State"), parse_str(ocr_value(expected_patient.get("Patient_State"))), mismatches)
                eob_eq(f"{pt}.Patient_Zip", db_patient.get("Patient_Zip"), parse_str(ocr_value(expected_patient.get("Patient_Zip"))), mismatches)
                eob_eq(f"{pt}.Patient_DOB", to_date_db(db_patient.get("Patient_DOB")), parse_date(ocr_value(expected_patient.get("Patient_DOB"))), mismatches)
                eob_eq(f"{pt}.Patient_Gender", db_patient.get("Patient_Gender"), parse_str(ocr_value(expected_patient.get("Patient_Gender"))), mismatches)
                eob_eq(f"{pt}.Patient_Relationship", db_patient.get("Patient_Relationship"), parse_str(ocr_value(expected_patient.get("Patient_Relationship"))), mismatches)
                eob_eq(f"{pt}.Insured_Name", db_patient.get("Insured_Name"), parse_str(ocr_value(expected_patient.get("Insured_Name"))), mismatches)
        elif db_patient is not None:
            eob_add_mismatch(f"{p}.Patient present in DB but missing in rawJson.", mismatches)

        expected_dx = claim_obj.get("Claim_Diagnosis")
        db_dx = fetch_rows_by_fk(conn, tables["diagnosis"], "ClaimId", db_claim["Id"], order_by="Id")
        if isinstance(expected_dx, dict):
            if not db_dx:
                eob_add_mismatch(f"{p}.Claim_Diagnosis missing in DB (expected present).", mismatches)
            else:
                dx = db_dx[0]
                pd = f"{p}.Claim_Diagnosis"
                eob_eq(f"{pd}.Primary_DX", dx.get("Primary_DX"), parse_str(ocr_value(expected_dx.get("Primary_DX"))), mismatches)
                for i in range(1, 13):
                    key = f"Secondary_DX{i}"
                    eob_eq(f"{pd}.{key}", dx.get(key), parse_str(ocr_value(expected_dx.get(key))), mismatches)
        elif db_dx:
            eob_add_mismatch(f"{p}.Claim_Diagnosis present in DB but missing in rawJson.", mismatches)

        expected_sls = claim_obj.get("Service_Line_Items") or []
        db_sls = fetch_rows_by_fk(conn, tables["service_line"], "ClaimId", db_claim["Id"], order_by="Id")
        if len(db_sls) != len(expected_sls):
            eob_add_mismatch(f"{p}.Service_Line_Items count mismatch. expected={len(expected_sls)}, actual={len(db_sls)}", mismatches)

        for sl_idx, (db_sl, sl_obj) in enumerate(zip(db_sls, expected_sls)):
            ps = f"{p}.Service_Line_Items[{sl_idx}]"
            eob_eq(f"{ps}.Service_From_Date", to_date_db(db_sl.get("Service_From_Date")), parse_date(ocr_value(sl_obj.get("Service_From_Date"))), mismatches)
            eob_eq(f"{ps}.Service_To_Date", to_date_db(db_sl.get("Service_To_Date")), parse_date(ocr_value(sl_obj.get("Service_To_Date"))), mismatches)
            eob_eq(f"{ps}.Procedure_Code", db_sl.get("Procedure_Code"), parse_str(ocr_value(sl_obj.get("Procedure_Code"))), mismatches)
            eob_eq(f"{ps}.Mod1", db_sl.get("Mod1"), parse_str(ocr_value(sl_obj.get("Mod1"))), mismatches)
            eob_eq(f"{ps}.Mod2", db_sl.get("Mod2"), parse_str(ocr_value(sl_obj.get("Mod2"))), mismatches)
            eob_eq(f"{ps}.Mod3", db_sl.get("Mod3"), parse_str(ocr_value(sl_obj.get("Mod3"))), mismatches)
            eob_eq(f"{ps}.Mod4", db_sl.get("Mod4"), parse_str(ocr_value(sl_obj.get("Mod4"))), mismatches)
            eob_eq(f"{ps}.Service_Billed_Amt", to_decimal_db(db_sl.get("Service_Billed_Amt")), parse_money(ocr_value(sl_obj.get("Service_Billed_Amt"))), mismatches)
            eob_eq(f"{ps}.Service_Allowed_Amt", to_decimal_db(db_sl.get("Service_Allowed_Amt")), parse_money(ocr_value(sl_obj.get("Service_Allowed_Amt"))), mismatches)
            eob_eq(f"{ps}.Service_Paid_Amt", to_decimal_db(db_sl.get("Service_Paid_Amt")), parse_money(ocr_value(sl_obj.get("Service_Paid_Amt"))), mismatches)
            eob_eq(f"{ps}.D_U", db_sl.get("D_U"), parse_str(ocr_value(sl_obj.get("D_U"))), mismatches)
            eob_eq(f"{ps}.Place_Of_Service", db_sl.get("Place_Of_Service"), parse_str(ocr_value(sl_obj.get("Place_Of_Service"))), mismatches)
            eob_eq(f"{ps}.DX_1", db_sl.get("DX_1"), parse_str(ocr_value(sl_obj.get("DX_1"))), mismatches)
            eob_eq(f"{ps}.DX_2", db_sl.get("DX_2"), parse_str(ocr_value(sl_obj.get("DX_2"))), mismatches)
            eob_eq(f"{ps}.DX_3", db_sl.get("DX_3"), parse_str(ocr_value(sl_obj.get("DX_3"))), mismatches)
            eob_eq(f"{ps}.DX_4", db_sl.get("DX_4"), parse_str(ocr_value(sl_obj.get("DX_4"))), mismatches)
            eob_eq(f"{ps}.User_Status", db_sl.get("User_Status"), parse_str(ocr_value(sl_obj.get("User_Status"))), mismatches)

            expected_adj = sl_obj.get("Service_Adjustments") or []
            db_adj = fetch_rows_by_fk(conn, tables["service_adjustment"], "ServiceLineItemId", db_sl["Id"], order_by="Id")
            if len(db_adj) != len(expected_adj):
                eob_add_mismatch(f"{ps}.Service_Adjustments count mismatch. expected={len(expected_adj)}, actual={len(db_adj)}", mismatches)
            for a_idx, (db_a, a_obj) in enumerate(zip(db_adj, expected_adj)):
                pa = f"{ps}.Service_Adjustments[{a_idx}]"
                eob_eq(f"{pa}.Service_Adjustment_Reason_Code", db_a.get("Service_Adjustment_Reason_Code"), parse_str(ocr_value(a_obj.get("Service_Adjustment_Reason_Code"))), mismatches)
                eob_eq(f"{pa}.Service_Adjustment_Group_Code", db_a.get("Service_Adjustment_Group_Code"), parse_str(ocr_value(a_obj.get("Service_Adjustment_Group_Code"))), mismatches)
                eob_eq(f"{pa}.Service_Adjustment_Reason", db_a.get("Service_Adjustment_Reason"), parse_str(ocr_value(a_obj.get("Service_Adjustment_Reason"))), mismatches)
                eob_eq(f"{pa}.Service_Adjustment_Amount", to_decimal_db(db_a.get("Service_Adjustment_Amount")), parse_money(ocr_value(a_obj.get("Service_Adjustment_Amount"))), mismatches)

    return mismatches

# ---------------------------------------------------------------------------
# Superbill Audit Core Comparison Functions
# ---------------------------------------------------------------------------

def sb_add_mismatch(message: str, mismatches: list[str]) -> None:
    if should_ignore_mismatch(message):
        return
    mismatches.append(message)
    active_stats = getattr(_SB_ACTIVE_STATS, "value", None)
    if active_stats is not None:
        active_stats.mismatches += 1

def sb_eq(path: str, actual: Any, expected: Any, mismatches: list[str]) -> None:
    active_stats = getattr(_SB_ACTIVE_STATS, "value", None)
    if actual == expected:
        if active_stats is not None:
            active_stats.matches += 1
        return
    if isinstance(actual, str) and isinstance(expected, str):
        actual_norm = normalize_compare_str(actual)
        expected_norm = normalize_compare_str(expected)
        if actual_norm == expected_norm:
            if active_stats is not None:
                active_stats.matches += 1
            return
        if is_diagnosis_path(path) and alnum_only(actual_norm) == alnum_only(expected_norm):
            if active_stats is not None:
                active_stats.matches += 1
            return
        if is_phone_path(path) and digits_only(actual_norm) == digits_only(expected_norm):
            if active_stats is not None:
                active_stats.matches += 1
            return
        if is_zip_path(path) and digits_only(actual_norm) == digits_only(expected_norm):
            if active_stats is not None:
                active_stats.matches += 1
            return
    sb_add_mismatch(f"{path} mismatch. expected={value_str(expected)}, actual={value_str(actual)}", mismatches)

def audit_superbill_allocation(
    allocation_row: dict[str, Any],
    conn: pymysql.connections.Connection,
    tables: dict[str, str],
) -> tuple[list[str], CompareStats]:
    mismatches: list[str] = []
    stats = CompareStats()
    _SB_ACTIVE_STATS.value = stats

    raw_json = allocation_row.get("RawJson") or allocation_row.get("rawJson")
    if not raw_json:
        sb_add_mismatch("RawJson missing on allocation row.", mismatches)
        _SB_ACTIVE_STATS.value = None
        return mismatches, stats

    try:
        root = json.loads(raw_json)
    except json.JSONDecodeError as ex:
        sb_add_mismatch(f"RawJson parse error: {ex}", mismatches)
        _SB_ACTIVE_STATS.value = None
        return mismatches, stats

    allocation = root.get("Allocation")
    if not isinstance(allocation, dict):
        sb_add_mismatch("RawJson does not contain Allocation object.", mismatches)
        _SB_ACTIVE_STATS.value = None
        return mismatches, stats

    sb_eq("Allocation.File_name", allocation_row.get("File_name"), allocation.get("File_name"), mismatches)
    sb_eq("Allocation.File_url", allocation_row.get("File_url"), allocation.get("File_url"), mismatches)
    sb_eq("Allocation.Client", allocation_row.get("Client"), allocation.get("Client"), mismatches)
    sb_eq("Allocation.Account", allocation_row.get("Account"), allocation.get("Account"), mismatches)
    sb_eq("Allocation.Total_No_Of_Clm_On_File", allocation_row.get("Total_No_Of_Clm_On_File"), len(allocation.get("Claim_Info") or []), mismatches)
    sb_eq("Allocation.Total_Charge_Amt_On_File", to_decimal_db(allocation_row.get("Total_Charge_Amt_On_File")), parse_money(allocation.get("Total_Charge_Amt_On_File")), mismatches)
    sb_eq("Allocation.Date_Of_Service", to_date_db(allocation_row.get("Date_Of_Service")), parse_date(allocation.get("Date_Of_Service")), mismatches)
    sb_eq("Allocation.Download_Date", to_date_db(allocation_row.get("Download_Date")), parse_date(allocation.get("Download_Date")), mismatches)
    sb_eq("Allocation.Completed_Date", to_date_db(allocation_row.get("Completed_Date")), parse_date(allocation.get("Completed_Date")), mismatches)
    sb_eq("Allocation.Not_Completed_Reason", allocation_row.get("Not_Completed_Reason"), allocation.get("Not_Completed_Reason"), mismatches)
    sb_eq("Allocation.File_Status", normalize_enum(allocation_row.get("File_Status")), map_file_status(allocation.get("File_Status")), mismatches)

    claim_wrappers = allocation.get("Claim_Info") or []
    expected_claims: list[dict[str, Any]] = []
    for wrapper in claim_wrappers:
        if isinstance(wrapper, dict) and isinstance(wrapper.get("Claim"), dict):
            expected_claims.append(wrapper["Claim"])

    db_claims = fetch_rows_by_fk(conn, tables["claim"], "AllocationRecordId", allocation_row["Id"], order_by="Id")
    if len(db_claims) != len(expected_claims):
        sb_add_mismatch(f"Claims count mismatch. expected={len(expected_claims)}, actual={len(db_claims)}", mismatches)

    for idx, (db_claim, claim_obj) in enumerate(zip(db_claims, expected_claims)):
        prefix = f"Claim[{idx}]"
        sb_eq(f"{prefix}.Patient_Control_Number", db_claim.get("Patient_Control_Number"), parse_str(ocr_value(claim_obj.get("Patient_Control_Number"))), mismatches)
        sb_eq(f"{prefix}.Claim_Total_Charge_Amt", to_decimal_db(db_claim.get("Claim_Total_Charge_Amt")), parse_money(ocr_value(claim_obj.get("Claim_Total_Charge_Amt"))), mismatches)
        sb_eq(f"{prefix}.Claim_Filing_indicator", db_claim.get("Claim_Filing_indicator"), parse_str(ocr_value(claim_obj.get("Claim_Filing_indicator"))), mismatches)
        sb_eq(f"{prefix}.Claim_Frequency_Code", db_claim.get("Claim_Frequency_Code"), parse_str(ocr_value(claim_obj.get("Claim_Frequency_Code"))), mismatches)
        sb_eq(f"{prefix}.Claim_Date_of_Service", to_date_db(db_claim.get("Claim_Date_of_Service")), parse_date(ocr_value(claim_obj.get("Claim_Date_of_Service"))), mismatches)
        sb_eq(f"{prefix}.Claim_Auth_No", db_claim.get("Claim_Auth_No"), parse_str(ocr_value(claim_obj.get("Claim_Auth_No"))), mismatches)
        sb_eq(f"{prefix}.Patient_Paid_Amt", to_decimal_db(db_claim.get("Patient_Paid_Amt")), parse_money(ocr_value(claim_obj.get("Patient_Paid_Amt"))), mismatches)
        sb_eq(f"{prefix}.CLIA_Number", db_claim.get("CLIA_Number"), parse_str(ocr_value(claim_obj.get("CLIA_Number"))), mismatches)
        sb_eq(f"{prefix}.CLaim_Invoice_Number", db_claim.get("CLaim_Invoice_Number"), parse_str(ocr_value(claim_obj.get("CLaim_Invoice_Number"))), mismatches)
        sb_eq(f"{prefix}.Admission_Date", to_date_db(db_claim.get("Admission_Date")), parse_date(ocr_value(claim_obj.get("Admission_Date"))), mismatches)
        sb_eq(f"{prefix}.Patient_Last_Seen_Date", to_date_db(db_claim.get("Patient_Last_Seen_Date")), parse_date(ocr_value(claim_obj.get("Patient_Last_Seen_Date"))), mismatches)

        expected_patient = claim_obj.get("Patient")
        db_patient = fetch_row_by_id(conn, tables["patient"], db_claim.get("PatientId")) if db_claim.get("PatientId") else None
        if isinstance(expected_patient, dict):
            if db_patient is None:
                sb_add_mismatch(f"{prefix}.Patient missing in DB (expected present).", mismatches)
            else:
                p = f"{prefix}.Patient"
                sb_eq(f"{p}.Patient_FN", db_patient.get("Patient_FN"), parse_str(ocr_value(expected_patient.get("Patient_FN"))), mismatches)
                sb_eq(f"{p}.Patient_LN", db_patient.get("Patient_LN"), parse_str(ocr_value(expected_patient.get("Patient_LN"))), mismatches)
                sb_eq(f"{p}.Patient_MI", db_patient.get("Patient_MI"), parse_str(ocr_value(expected_patient.get("Patient_MI"))), mismatches)
                sb_eq(f"{p}.Patient_Id", db_patient.get("Patient_Id"), parse_str(ocr_value(expected_patient.get("Patient_Id"))), mismatches)
                sb_eq(f"{p}.Patient_Account_Number", db_patient.get("Patient_Account_Number"), parse_str(ocr_value(expected_patient.get("Patient_Account_Number"))), mismatches)
                sb_eq(f"{p}.Patient_Control_Number", db_patient.get("Patient_Control_Number"), parse_str(ocr_value(expected_patient.get("Patient_Control_Number"))), mismatches)
                sb_eq(f"{p}.Patient_Group", db_patient.get("Patient_Group"), parse_str(ocr_value(expected_patient.get("Patient_Group"))), mismatches)
                sb_eq(f"{p}.Patient_Addr1", db_patient.get("Patient_Addr1"), parse_str(ocr_value(expected_patient.get("Patient_Addr1"))), mismatches)
                sb_eq(f"{p}.Patient_Addr2", db_patient.get("Patient_Addr2"), parse_str(ocr_value(expected_patient.get("Patient_Addr2"))), mismatches)
                sb_eq(f"{p}.Patient_City", db_patient.get("Patient_City"), parse_str(ocr_value(expected_patient.get("Patient_City"))), mismatches)
                sb_eq(f"{p}.Patient_State", db_patient.get("Patient_State"), parse_str(ocr_value(expected_patient.get("Patient_State"))), mismatches)
                sb_eq(f"{p}.Patient_Zip", db_patient.get("Patient_Zip"), parse_str(ocr_value(expected_patient.get("Patient_Zip"))), mismatches)
                sb_eq(f"{p}.Patient_DOB", to_date_db(db_patient.get("Patient_DOB")), parse_date(ocr_value(expected_patient.get("Patient_DOB"))), mismatches)
                sb_eq(f"{p}.Patient_Gender", db_patient.get("Patient_Gender"), parse_str(ocr_value(expected_patient.get("Patient_Gender"))), mismatches)
                sb_eq(f"{p}.Patient_Relationship", db_patient.get("Patient_Relationship"), parse_str(ocr_value(expected_patient.get("Patient_Relationship"))), mismatches)
                sb_eq(f"{p}.Patient_Marital_Status", db_patient.get("Patient_Marital_Status"), parse_str(ocr_value(expected_patient.get("Patient_Marital_Status"))), mismatches)
                sb_eq(f"{p}.Patient_Primary_Phone_No", db_patient.get("Patient_Primary_Phone_No"), parse_str(ocr_value(expected_patient.get("Patient_Primary_Phone_No"))), mismatches)
                sb_eq(f"{p}.Patient_Home_Phone_No", db_patient.get("Patient_Home_Phone_No"), parse_str(ocr_value(expected_patient.get("Patient_Home_Phone_No"))), mismatches)
                sb_eq(f"{p}.Patient_Primary_Email", db_patient.get("Patient_Primary_Email"), parse_str(ocr_value(expected_patient.get("Patient_Primary_Email"))), mismatches)
                sb_eq(f"{p}.Insured_Name", db_patient.get("Insured_Name"), parse_str(ocr_value(expected_patient.get("Insured_Name"))), mismatches)
        elif db_patient is not None:
            sb_add_mismatch(f"{prefix}.Patient present in DB but missing in RawJson.", mismatches)

        expected_payers = claim_obj.get("Payer") or []
        db_payers = fetch_rows_by_fk(conn, tables["payer"], "ClaimRecordId", db_claim["Id"], order_by="Id")
        if len(db_payers) != len(expected_payers):
            sb_add_mismatch(f"{prefix}.Payer count mismatch. expected={len(expected_payers)}, actual={len(db_payers)}", mismatches)
        for p_idx, (db_payer, payer_obj) in enumerate(zip(db_payers, expected_payers)):
            pp = f"{prefix}.Payer[{p_idx}]"
            sb_eq(f"{pp}.Payer_Name", db_payer.get("Payer_Name"), parse_str(ocr_value((payer_obj or {}).get("Payer_Name"))), mismatches)
            sb_eq(f"{pp}.Payer_Id", db_payer.get("Payer_Id"), parse_str(ocr_value((payer_obj or {}).get("Payer_Id"))), mismatches)
            sb_eq(f"{pp}.Payer_Addr1", db_payer.get("Payer_Addr1"), parse_str(ocr_value((payer_obj or {}).get("Payer_Addr1"))), mismatches)
            sb_eq(f"{pp}.Payer_Addr2", db_payer.get("Payer_Addr2"), parse_str(ocr_value((payer_obj or {}).get("Payer_Addr2"))), mismatches)
            sb_eq(f"{pp}.Payer_City", db_payer.get("Payer_City"), parse_str(ocr_value((payer_obj or {}).get("Payer_City"))), mismatches)
            sb_eq(f"{pp}.Payer_State", db_payer.get("Payer_State"), parse_str(ocr_value((payer_obj or {}).get("Payer_State"))), mismatches)
            sb_eq(f"{pp}.Payer_Zip", db_payer.get("Payer_Zip"), parse_str(ocr_value((payer_obj or {}).get("Payer_Zip"))), mismatches)
            sb_eq(f"{pp}.Payer_Type", db_payer.get("Payer_Type"), parse_str(ocr_value((payer_obj or {}).get("Payer_Type"))), mismatches)

        expected_provider_items = []
        provider_roles = [
            ("BillingProvider", "BillingProvider", False),
            ("ServicingProvider", "ServicingProvider", False),
            ("ReferringProvider", "ReferringProvider", False),
            ("OrderingProvider", "OrderingProvider", False),
            ("SupervisingProvider", "SupervisingProvider", False),
            ("ServiceFacilityProvider", "ServicingFacility", True),
        ]
        for role_name, json_key, is_facility in provider_roles:
            obj = claim_obj.get(json_key)
            if isinstance(obj, dict):
                expected_provider_items.append((role_name, obj, is_facility))

        db_providers = fetch_rows_by_fk(conn, tables["provider"], "ClaimRecordId", db_claim["Id"], order_by="Id")
        if len(db_providers) != len(expected_provider_items):
            sb_add_mismatch(f"{prefix}.Provider count mismatch. expected={len(expected_provider_items)}, actual={len(db_providers)}", mismatches)

        db_by_role = {}
        for row_prov in db_providers:
            role = normalize_enum(row_prov.get("Role")) or "unknown"
            db_by_role.setdefault(role, []).append(row_prov)

        exp_by_role = {}
        for role, obj, is_facility in expected_provider_items:
            exp_by_role.setdefault(role, []).append((obj, is_facility))

        for role, exp_list in exp_by_role.items():
            db_list = db_by_role.get(role, [])
            if len(db_list) != len(exp_list):
                sb_add_mismatch(f"{prefix}.Provider role={role} count mismatch. expected={len(exp_list)}, actual={len(db_list)}", mismatches)
            for r_idx, (db_row, (exp_obj, is_facility)) in enumerate(zip(db_list, exp_list)):
                rp = f"{prefix}.Provider[{role}][{r_idx}]"
                if is_facility:
                    sb_eq(f"{rp}.Provider_Name", db_row.get("Provider_Name"), parse_str(ocr_value(exp_obj.get("Facility_Name"))), mismatches)
                    sb_eq(f"{rp}.Provider_NPI", db_row.get("Provider_NPI"), parse_str(ocr_value(exp_obj.get("Facility_NPI"))), mismatches)
                    sb_eq(f"{rp}.Provider_Addr1", db_row.get("Provider_Addr1"), parse_str(ocr_value(exp_obj.get("Facility_Addr1"))), mismatches)
                    sb_eq(f"{rp}.Provider_Addr2", db_row.get("Provider_Addr2"), parse_str(ocr_value(exp_obj.get("Facility_Addr2"))), mismatches)
                    sb_eq(f"{rp}.Provider_City", db_row.get("Provider_City"), parse_str(ocr_value(exp_obj.get("Provider_City"))), mismatches)
                    sb_eq(f"{rp}.Provider_State", db_row.get("Provider_State"), parse_str(ocr_value(exp_obj.get("Facility_State"))), mismatches)
                    sb_eq(f"{rp}.Provider_Zip", db_row.get("Provider_Zip"), parse_str(ocr_value(exp_obj.get("Facility_Zip"))), mismatches)
                    sb_eq(f"{rp}.Provider_FedId", db_row.get("Provider_FedId"), parse_str(ocr_value(exp_obj.get("Facility_FedId"))), mismatches)
                    sb_eq(f"{rp}.Provider_TaxId", db_row.get("Provider_TaxId"), parse_str(ocr_value(exp_obj.get("Facility_TaxId"))), mismatches)
                    sb_eq(f"{rp}.Provider_Taxonomy", db_row.get("Provider_Taxonomy"), parse_str(ocr_value(exp_obj.get("Facility_Taxonomy"))), mismatches)
                else:
                    sb_eq(f"{rp}.Provider_Name", db_row.get("Provider_Name"), parse_str(ocr_value(exp_obj.get("Provider_Name"))), mismatches)
                    sb_eq(f"{rp}.Provider_NPI", db_row.get("Provider_NPI"), parse_str(ocr_value(exp_obj.get("Provider_NPI"))), mismatches)
                    sb_eq(f"{rp}.Provider_Addr1", db_row.get("Provider_Addr1"), parse_str(ocr_value(exp_obj.get("Provider_Addr1"))), mismatches)
                    sb_eq(f"{rp}.Provider_Addr2", db_row.get("Provider_Addr2"), parse_str(ocr_value(exp_obj.get("Provider_Addr2"))), mismatches)
                    sb_eq(f"{rp}.Provider_City", db_row.get("Provider_City"), parse_str(ocr_value(exp_obj.get("Provider_City"))), mismatches)
                    sb_eq(f"{rp}.Provider_State", db_row.get("Provider_State"), parse_str(ocr_value(exp_obj.get("Provider_State"))), mismatches)
                    sb_eq(f"{rp}.Provider_Zip", db_row.get("Provider_Zip"), parse_str(ocr_value(exp_obj.get("Provider_Zip"))), mismatches)
                    sb_eq(f"{rp}.Provider_FedId", db_row.get("Provider_FedId"), parse_str(ocr_value(exp_obj.get("Provider_FedId"))), mismatches)
                    sb_eq(f"{rp}.Provider_TaxId", db_row.get("Provider_TaxId"), parse_str(ocr_value(exp_obj.get("Provider_TaxId"))), mismatches)
                    sb_eq(f"{rp}.Provider_Taxonomy", db_row.get("Provider_Taxonomy"), parse_str(ocr_value(exp_obj.get("Provider_Taxonomy"))), mismatches)

        expected_dx = claim_obj.get("ClaimDiagnosisCodes")
        db_dx = fetch_rows_by_fk(conn, tables["dx"], "ClaimRecordId", db_claim["Id"], order_by="Id")
        if isinstance(expected_dx, dict):
            if len(db_dx) == 0:
                sb_add_mismatch(f"{prefix}.Diagnosis missing in DB (expected present).", mismatches)
            else:
                dx_row = db_dx[0]
                dp = f"{prefix}.Diagnosis"
                sb_eq(f"{dp}.Primary_DX", dx_row.get("Primary_DX"), parse_str(ocr_value(expected_dx.get("Primary_DX"))), mismatches)
                for i in range(1, 13):
                    key = f"Secondary_DX{i}"
                    sb_eq(f"{dp}.{key}", dx_row.get(key), parse_str(ocr_value(expected_dx.get(key))), mismatches)
        elif len(db_dx) > 0:
            sb_add_mismatch(f"{prefix}.Diagnosis present in DB but missing in RawJson.", mismatches)

        expected_sls = claim_obj.get("ServiceLines") or []
        db_sls = fetch_rows_by_fk(conn, tables["sl"], "ClaimRecordId", db_claim["Id"], order_by="Id")
        if len(db_sls) != len(expected_sls):
            sb_add_mismatch(f"{prefix}.ServiceLines count mismatch. expected={len(expected_sls)}, actual={len(db_sls)}", mismatches)
        for s_idx, (db_sl, sl_obj) in enumerate(zip(db_sls, expected_sls)):
            sp = f"{prefix}.ServiceLines[{s_idx}]"
            sb_eq(f"{sp}.Service_From_Date", to_date_db(db_sl.get("Service_From_Date")), parse_date(ocr_value((sl_obj or {}).get("Service_From_Date"))), mismatches)
            sb_eq(f"{sp}.Service_To_Date", to_date_db(db_sl.get("Service_To_Date")), parse_date(ocr_value((sl_obj or {}).get("Service_To_Date"))), mismatches)
            sb_eq(f"{sp}.Procedure_Code", db_sl.get("Procedure_Code"), parse_str(ocr_value((sl_obj or {}).get("Procedure_Code"))), mismatches)
            sb_eq(f"{sp}.Mod1", db_sl.get("Mod1"), parse_str(ocr_value((sl_obj or {}).get("Mod1"))), mismatches)
            sb_eq(f"{sp}.Mod2", db_sl.get("Mod2"), parse_str(ocr_value((sl_obj or {}).get("Mod2"))), mismatches)
            sb_eq(f"{sp}.Mod3", db_sl.get("Mod3"), parse_str(ocr_value((sl_obj or {}).get("Mod3"))), mismatches)
            sb_eq(f"{sp}.Mod4", db_sl.get("Mod4"), parse_str(ocr_value((sl_obj or {}).get("Mod4"))), mismatches)
            sb_eq(f"{sp}.Service_Billed_Amt", to_decimal_db(db_sl.get("Service_Billed_Amt")), parse_money(ocr_value((sl_obj or {}).get("Service_Billed_Amt"))), mismatches)
            sb_eq(f"{sp}.D_U", db_sl.get("D_U"), parse_int(ocr_value_any((sl_obj or {}), "D_U", "D/U", "DU")), mismatches)
            sb_eq(f"{sp}.Place_Of_Service", db_sl.get("Place_Of_Service"), parse_str(ocr_value((sl_obj or {}).get("Place_Of_Service"))), mismatches)
            sb_eq(f"{sp}.DX_1", db_sl.get("DX_1"), parse_str(ocr_value((sl_obj or {}).get("DX_1"))), mismatches)
            sb_eq(f"{sp}.DX_2", db_sl.get("DX_2"), parse_str(ocr_value((sl_obj or {}).get("DX_2"))), mismatches)
            sb_eq(f"{sp}.DX_3", db_sl.get("DX_3"), parse_str(ocr_value((sl_obj or {}).get("DX_3"))), mismatches)
            sb_eq(f"{sp}.DX_4", db_sl.get("DX_4"), parse_str(ocr_value((sl_obj or {}).get("DX_4"))), mismatches)
            sb_eq(f"{sp}.User_Status", db_sl.get("User_Status"), parse_str(ocr_value((sl_obj or {}).get("User_Status"))), mismatches)

    _SB_ACTIVE_STATS.value = None
    return mismatches, stats


# ---------------------------------------------------------------------------
# Langfuse Helpers
# ---------------------------------------------------------------------------

def get_azure_setting(key: str, default=None):
    """Reads from local.settings.json Values first, then falls back to env vars."""
    try:
        if os.path.exists("local.settings.json"):
            with open("local.settings.json", "r") as f:
                settings = json.load(f)
                if "Values" in settings and key in settings["Values"]:
                    return settings["Values"][key]
    except Exception as e:
        logging.warning(f"Could not read local.settings.json: {e}")
    return os.getenv(key, default)


def _parse_date_from_filename(file_name: str) -> dt.date | None:
    """Extracts YYYYMMDD date embedded in filenames like *_20260619112812_*.pdf"""
    m = re.search(r"(\d{8})\d{6}", file_name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    return None


def _parse_superbill_date_from_filename(file_name: str) -> dt.date | None:
    """
    Extracts date from superbill filenames which use MMDDYYYY format
    e.g. 'Jennifer Soyama 04202026 - Coded anitha 04212026.pdf' -> 2026-04-20
    Tries MMDDYYYY first, then YYYYMMDD as fallback.
    Also handles filenames where the date is embedded at the start of a longer
    digit sequence, e.g. '20260702065403746939' -> 2026-07-02.
    """
    # Match 8 digits not preceded by another digit (handles both standalone and
    # prefix-of-longer-sequence cases, e.g. 20260702 inside 20260702065403...)
    for m in re.finditer(r"(?<!\d)(\d{8})", file_name):
        token = m.group(1)
        # Try MMDDYYYY (superbill convention)
        try:
            return datetime.strptime(token, "%m%d%Y").date()
        except ValueError:
            pass
        # Try YYYYMMDD (tabak convention)
        try:
            return datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            pass
    return None


def fetch_superbill_langfuse_metrics(file_name: str, langfuse_client: Langfuse) -> Dict[str, Any]:
    """
    Searches Medical-Billing-Pipeline traces matching the file_name in Langfuse
    to retrieve token counts, total cost, and latency.
    Narrows the search window using the date embedded in the filename.
    Token counts come from child observations since trace.usage may be None.
    """
    metrics = {"input_token": 0, "output_token": 0, "cost": 0.0, "latency": 0.0}

    if not langfuse_client or not file_name:
        return metrics

    try:
        file_date = _parse_superbill_date_from_filename(file_name)
        if file_date:
            from_ts = datetime(file_date.year, file_date.month, file_date.day,
                               tzinfo=dt.timezone.utc)
            to_ts = from_ts + dt.timedelta(days=4)
        else:
            from_ts = datetime.now(dt.timezone.utc) - dt.timedelta(days=90)
            to_ts = None

        matched_trace_id = None
        page = 1
        while page <= 10 and matched_trace_id is None:
            fetch_kwargs = dict(name="Medical-Billing-Pipeline", from_timestamp=from_ts, limit=100, page=page)
            if to_ts:
                fetch_kwargs["to_timestamp"] = to_ts

            result = langfuse_client.fetch_traces(**fetch_kwargs)
            traces = result.data if result and result.data else []
            logging.info(f"[Langfuse] page={page} traces_returned={len(traces)} for file='{file_name}' window=[{from_ts}, {to_ts}]")
            if not traces:
                break

            for t in traces:
                inp = t.input or {}
                try:
                    args = inp.get("args") or []
                    first_arg = args[0] if args else {}
                    # Superbill traces use args[0].name; also try filename/file_name
                    trace_file = (
                        first_arg.get("name")
                        or first_arg.get("filename")
                        or first_arg.get("file_name")
                        or inp.get("filename")
                        or inp.get("file_name")
                    )
                except Exception:
                    trace_file = None

                if page == 1 and traces.index(t) == 0:
                    # Log the first trace's name and input structure to diagnose mismatches
                    logging.info(f"[Langfuse] First trace: id={t.id}, name={getattr(t, 'name', None)!r}, input_keys={list(inp.keys()) if isinstance(inp, dict) else type(inp).__name__}, trace_file={trace_file!r}")

                if trace_file and trace_file == file_name:
                    metrics["cost"] = float(getattr(t, "total_cost", 0.0) or 0.0)
                    metrics["latency"] = float(getattr(t, "latency", 0.0) or 0.0)
                    matched_trace_id = t.id
                    break

            if len(traces) < 100:
                break
            page += 1

        # Trace.usage is None — get tokens from child observations
        if matched_trace_id:
            try:
                obs_result = langfuse_client.fetch_observations(trace_id=matched_trace_id, limit=50)
                for obs in (obs_result.data or []):
                    usage = getattr(obs, "usage", None)
                    if usage:
                        metrics["input_token"] += int(getattr(usage, "input", 0) or 0)
                        metrics["output_token"] += int(getattr(usage, "output", 0) or 0)
            except Exception as e:
                logging.warning(f"Failed to fetch observations for trace {matched_trace_id}: {e}")

    except Exception as e:
        # Re-raise network/timeout errors so the caller can short-circuit remaining files.
        # All other errors are swallowed and 0s are returned.
        _err_lower = str(e).lower()
        if "timed out" in _err_lower or "connection" in _err_lower or "network" in _err_lower:
            raise
        logging.warning(f"Failed to fetch Langfuse metrics for '{file_name}': {e}")

    return metrics


# ---------------------------------------------------------------------------
# Core Entry Point Processors
# ---------------------------------------------------------------------------

def process_accuracy_healthcare_eob(payload: dict):
    logging.info("[f2 Healthcare EOB Accuracy] Processing payload...")
    record_ids = payload.get("record_ids") or payload.get("id") or []
    environment = payload.get("environment") or get_environment() or "exp"

    if not record_ids:
        logging.warning("[f2 Healthcare EOB Accuracy] No record_ids or id list provided.")
        return

    try:
        cfg = resolve_db_config()
    except Exception as e:
        logging.error(f"[f2 Healthcare EOB Accuracy] Failed to resolve DB config: {e}", exc_info=True)
        return

    try:
        conn = pymysql.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            charset="utf8mb4"
        )
    except Exception as e:
        logging.error(f"[f2 Healthcare EOB Accuracy] Database connection failed: {e}", exc_info=True)
        return

    # -----------------------------------------------------------------------
    # All DB work (fetch + audit) inside a single connection block to avoid
    # InterfaceError caused by using conn after the transaction context exits.
    # audit_eob_allocation calls fetch_rows_by_fk/fetch_row_by_id which need
    # an open connection — they must run before conn closes.
    # -----------------------------------------------------------------------
    clickhouse_rows = []
    try:
        with conn:
            tables = {
                "allocation": resolve_table_name(conn, ["EOB_Allocation", "EOBAllocations"]),
                "claim": resolve_table_name(conn, ["EOB_Claim", "EOBClaims"]),
                "payer": resolve_table_name(conn, ["EOB_Payer", "EOBPayers"]),
                "payee": resolve_table_name(conn, ["EOB_Payee", "EOBPayees"]),
                "diagnosis": resolve_table_name(conn, ["EOB_Claim_Diagnosis", "EOBClaim_Diagnosis"]),
                "service_line": resolve_table_name(conn, ["EOB_Service_Line_Item", "EOBService_Line_Items"]),
                "service_adjustment": resolve_table_name(conn, ["EOB_Service_Adjustment", "EOBService_Adjustments"]),
                "patient": resolve_table_name(conn, ["PatientRecord", "PatientRecords"]),
            }

            placeholders = ", ".join(["%s"] * len(record_ids))
            query = f"SELECT * FROM `{tables['allocation']}` WHERE Id IN ({placeholders})"
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(record_ids))
                rows = cursor.fetchall()

            logging.info(f"[f2 Healthcare EOB Accuracy] Fetched {len(rows)} matching allocations.")

            for row in rows:
                alloc_id = str(row.get("Id"))

                global _EOB_ACTIVE_STATS
                _EOB_ACTIVE_STATS = CompareStats()

                try:
                    mismatches = audit_eob_allocation(row, conn, tables)
                    stats = _EOB_ACTIVE_STATS or CompareStats()
                except Exception as aud_ex:
                    logging.error(f"[f2 Healthcare EOB Accuracy] Audit failed for EOB ID {alloc_id}: {aud_ex}", exc_info=True)
                    continue
                finally:
                    _EOB_ACTIVE_STATS = None

                total_compared = stats.matches + len(mismatches)
                accuracy_pct = (stats.matches / total_compared * 100.0) if total_compared else 0.0

                clickhouse_rows.append({
                    "item_id": f"EOB::{alloc_id}",
                    "source_type": "EOB",
                    "allocation_id": alloc_id,
                    "date": _extract_date_string(row.get("Download_Date") or row.get("download_date")),
                    "date_time": _to_iso_datetime_string(row.get("UpdatedAt") or row.get("UpdatedOn") or row.get("CreatedOn")),
                    "file_name": row.get("File_name") or row.get("file_name") or "",
                    "client_name": row.get("Client") or row.get("client_name") or "",
                    "total_matched": stats.matches,
                    "total_mismatches": len(mismatches),
                    "accuracy": round(accuracy_pct, 4)
                })
    except Exception as e:
        logging.error(f"[f2 Healthcare EOB Accuracy] DB operation failed: {e}", exc_info=True)
        return

    logging.info(f"[f2 Healthcare EOB Accuracy] Audited {len(clickhouse_rows)} allocations.")

    if clickhouse_rows:
        try:
            inserted = insert_healthcare_accuracy_rows(HEALTHCARE_EOB_ACCURACY_TABLE, environment, clickhouse_rows)
            logging.info(f"[f2 Healthcare EOB Accuracy] Successfully stored {inserted} evaluation summaries in ClickHouse.")
        except Exception as ch_err:
            logging.error(f"[f2 Healthcare EOB Accuracy] Failed to insert summaries into ClickHouse: {ch_err}", exc_info=True)


def process_accuracy_healthcare_superbill(payload: dict):
    logging.info("[f2 Healthcare Superbill Accuracy] Processing payload...")
    record_ids = payload.get("record_ids") or payload.get("id") or []
    environment = payload.get("environment") or get_environment() or "exp"

    if not record_ids:
        logging.warning("[f2 Healthcare Superbill Accuracy] No record_ids or id list provided.")
        return

    try:
        cfg = resolve_db_config()
    except Exception as e:
        logging.error(f"[f2 Healthcare Superbill Accuracy] Failed to resolve DB config: {e}", exc_info=True)
        return

    try:
        conn = pymysql.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            charset="utf8mb4"
        )
    except Exception as e:
        logging.error(f"[f2 Healthcare Superbill Accuracy] Database connection failed: {e}", exc_info=True)
        return

    # -----------------------------------------------------------------------
    # All DB work (fetch + audit) inside a single connection block to avoid
    # InterfaceError caused by using conn after the transaction context exits
    # -----------------------------------------------------------------------
    audit_results = []
    try:
        with conn:
            tables = {
                "allocation": resolve_table_name(conn, ["AllocationRecord", "SuperBillAllocation", "SuperBillAllocations"]),
                "claim": resolve_table_name(conn, ["ClaimRecord", "SuperBillClaim", "SuperBillClaims"]),
                "payer": resolve_table_name(conn, ["PayerRecord", "SuperBillPayer", "SuperBillPayers"]),
                "provider": resolve_table_name(conn, ["ClaimProviderRecord", "SuperBillClaimProvider", "SuperBillClaimProviders"]),
                "patient": resolve_table_name(conn, ["PatientRecord", "PatientRecords"]),
                "dx": resolve_table_name(conn, ["DiagnosisCodesRecord", "SuperBillDiagnosisCode", "SuperBillDiagnosisCodes"]),
                "sl": resolve_table_name(conn, ["ServiceLineRecord", "SuperBillServiceLine", "SuperBillServiceLines"]),
            }

            placeholders = ", ".join(["%s"] * len(record_ids))
            query = f"SELECT * FROM `{tables['allocation']}` WHERE Id IN ({placeholders})"
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(record_ids))
                rows = cursor.fetchall()

            logging.info(f"[f2 Healthcare Superbill Accuracy] Fetched {len(rows)} matching allocations.")

            for row in rows:
                alloc_id = str(row.get("Id"))
                file_name = row.get("File_name") or row.get("file_name") or ""

                stats = CompareStats()
                _SB_ACTIVE_STATS.value = stats
                try:
                    mismatches, stats = audit_superbill_allocation(row, conn, tables)
                except Exception as aud_ex:
                    logging.error(f"[f2 Healthcare Superbill Accuracy] Audit failed for Superbill ID {alloc_id}: {aud_ex}", exc_info=True)
                    _SB_ACTIVE_STATS.value = None
                    continue
                finally:
                    _SB_ACTIVE_STATS.value = None

                total_compared = stats.matches + stats.mismatches
                accuracy_pct = (stats.matches / total_compared * 100.0) if total_compared else 0.0

                audit_results.append({
                    "alloc_id": alloc_id,
                    "file_name": file_name,
                    "date": _extract_date_string(row.get("Download_Date") or row.get("download_date")),
                    "date_time": _to_iso_datetime_string(row.get("UpdatedAt") or row.get("UpdatedOn") or row.get("CreatedOn")),
                    "client_name": row.get("Client") or row.get("client_name") or "",
                    "total_matched": stats.matches,
                    "total_mismatches": stats.mismatches,
                    "accuracy": round(accuracy_pct, 4),
                })
    except Exception as e:
        logging.error(f"[f2 Healthcare Superbill Accuracy] DB operation failed: {e}", exc_info=True)
        return

    logging.info(f"[f2 Healthcare Superbill Accuracy] Audited {len(audit_results)} allocations.")

    # Initialize Langfuse AFTER DB connection is released
    langfuse_client = None
    lf_public_key = get_azure_setting("HEALTHCARE_LANGFUSE_PUBLIC_KEY")
    lf_secret_key = get_azure_setting("HEALTHCARE_LANGFUSE_SECRET_KEY")
    lf_host = get_azure_setting("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if lf_public_key and lf_secret_key:
        try:
            _lf_kwargs = {}
            if _httpx is not None:
                _lf_kwargs["httpx_client"] = _httpx.Client(
                    timeout=_httpx.Timeout(connect=5.0, read=8.0, write=10.0, pool=5.0)
                )
            langfuse_client = Langfuse(public_key=lf_public_key, secret_key=lf_secret_key, host=lf_host, **_lf_kwargs)
            logging.info("[f2 Healthcare Superbill Accuracy] Langfuse client initialized (read timeout=8s).")
        except Exception as lf_err:
            logging.warning(f"[f2 Healthcare Superbill Accuracy] Langfuse init failed: {lf_err}")

    # Build ClickHouse rows from audit results first, using zeros for Langfuse fields.
    # Langfuse metrics are enriched below only when available — they must not block the insert.
    clickhouse_rows = []
    for result in audit_results:
        clickhouse_rows.append({
            "item_id": f"Superbill::{result['alloc_id']}",
            "source_type": "Superbill",
            "allocation_id": result["alloc_id"],
            "date": result["date"],
            "date_time": result["date_time"],
            "file_name": result["file_name"],
            "client_name": result["client_name"],
            "total_matched": result["total_matched"],
            "total_mismatches": result["total_mismatches"],
            "accuracy": result["accuracy"],
            "input_token": 0,
            "output_token": 0,
            "cost": 0.0,
            "latency": 0.0,
        })

    # Persist accuracy data to ClickHouse immediately — independent of Langfuse.
    if clickhouse_rows:
        try:
            inserted = insert_healthcare_accuracy_rows(HEALTHCARE_SUPERBILL_ACCURACY_TABLE, environment, clickhouse_rows)
            logging.info(f"[f2 Healthcare Superbill Accuracy] Successfully stored {inserted} evaluation summaries in ClickHouse.")
        except Exception as ch_err:
            logging.error(f"[f2 Healthcare Superbill Accuracy] Failed to insert summaries into ClickHouse: {ch_err}", exc_info=True)

    # Enrich with Langfuse metrics (best-effort).
    # If Langfuse is unreachable (timeout/connection error on first call), the loop is
    # immediately abandoned and the 0s already saved to ClickHouse are kept as-is.
    lf_enriched = []
    lf_unreachable = False
    for result in audit_results:
        if lf_unreachable:
            continue
        try:
            lf_metrics = fetch_superbill_langfuse_metrics(result["file_name"], langfuse_client)
        except Exception:
            lf_unreachable = True
            logging.warning(
                "[f2 Healthcare Superbill Accuracy] Langfuse unreachable — "
                "skipping enrichment for all remaining files, 0s are kept for token/cost/latency."
            )
            continue
        if lf_metrics["input_token"] or lf_metrics["output_token"] or lf_metrics["cost"] or lf_metrics["latency"]:
            lf_enriched.append({
            "item_id": f"Superbill::{result['alloc_id']}",
            "source_type": "Superbill",
            "allocation_id": result["alloc_id"],
            "date": result["date"],
            "date_time": result["date_time"],
            "file_name": result["file_name"],
            "client_name": result["client_name"],
            "total_matched": result["total_matched"],
            "total_mismatches": result["total_mismatches"],
            "accuracy": result["accuracy"],
            "input_token": lf_metrics["input_token"],
            "output_token": lf_metrics["output_token"],
            "cost": lf_metrics["cost"],
            "latency": lf_metrics["latency"],
        })

    if lf_enriched:
        try:
            insert_healthcare_accuracy_rows(HEALTHCARE_SUPERBILL_ACCURACY_TABLE, environment, lf_enriched)
            logging.info(f"[f2 Healthcare Superbill Accuracy] Updated {len(lf_enriched)} rows with Langfuse metrics in ClickHouse.")
        except Exception as ch_err:
            logging.error(f"[f2 Healthcare Superbill Accuracy] Failed to update Langfuse metrics in ClickHouse: {ch_err}", exc_info=True)
