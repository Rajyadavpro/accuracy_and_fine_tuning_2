from __future__ import annotations

import datetime as dt
import logging
import os
import requests
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional

# ---------------------------------------------------------------------------
# Logging setup (matches accuracy/clickhouse_http_store.py style)
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "AUX_code"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_push.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(_ch)

CLICKHOUSE_HOST = "CLICKHOUSE_HOST"
CLICKHOUSE_DATABASE = "CLICKHOUSE_DATABASE"
CLICKHOUSE_USER = "CLICKHOUSE_USER"
CLICKHOUSE_PASSWORD = "CLICKHOUSE_PASSWORD"
STORAGE_ENVIRONMENT = "STORAGE_ENVIRONMENT"

IDP_ACCURACY_SUMMARY_TABLE = "idp_accuracy_summary"
IDP_ACCURACY_CLIENT_TABLE = "idp_accuracy_client"
TABAK_ACCURACY_SUMMARY_TABLE = "tabak_accuracy_summary"
TABAK_ACCURACY_DAILY_TABLE = "tabak_accuracy_daily"
TABAK_ACCURACY_CATEGORY_TABLE = "tabak_accuracy_category"
TABAK_ACCURACY_SUBCATEGORY_TABLE = "tabak_accuracy_subcategory"
HEALTHCARE_EOB_ACCURACY_TABLE = "healthcare_accuracy_eob"
HEALTHCARE_SUPERBILL_ACCURACY_TABLE = "healthcare_accuracy_superbill"
IDP_FINETUNING_CHECKPOINT_TABLE = "idp_finetuning_checkpoint"
TABAK_FINETUNING_CHECKPOINT_TABLE = "tabak_finetuning_checkpoint"
EOB_FINETUNING_CHECKPOINT_TABLE = "eob_finetuning_checkpoint"
SUPERBILL_FINETUNING_CHECKPOINT_TABLE = "superbill_finetuning_checkpoint"
AUDIO_FINETUNING_CHECKPOINT_TABLE = "audio_finetuning_checkpoint"


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def get_environment() -> str:
    return _env(STORAGE_ENVIRONMENT) or _env("LANGFUSE_ENVIRONMENT") or "dev"


def _client_config() -> tuple[str, int, str, str, str]:
    host = _env(CLICKHOUSE_HOST, "172.173.148.33")
    http_port = int(_env("CLICKHOUSE_HTTP_PORT", "8123"))
    database = _env(CLICKHOUSE_DATABASE, "accuracy_and_finetuning")
    user = _env(CLICKHOUSE_USER, "admin")
    password = _env(CLICKHOUSE_PASSWORD, "Holly7583hfxZ")
    return host, http_port, database, user, password


# Alias used by EOB/Superbill accuracy scripts
_get_clickhouse_config = _client_config


def _http_exec(sql: str, timeout: int = 30) -> bool:
    """Execute a DDL/DML via HTTP POST."""
    host, http_port, database, user, password = _client_config()
    url = f"http://{host}:{http_port}/"
    resp = requests.post(url, auth=(user, password), data=sql.encode(), timeout=timeout)
    if resp.status_code not in (200, 201):
        print(f"[ClickHouse] HTTP exec failed {resp.status_code}: {resp.text[:300]}")
        return False
    return True


def _http_query(sql: str, timeout: int = 15) -> Optional[str]:
    """Run a SELECT and return raw text."""
    host, http_port, database, user, password = _client_config()
    url = f"http://{host}:{http_port}/"
    resp = requests.get(url, auth=(user, password), params={"query": sql}, timeout=timeout)
    if resp.status_code == 200:
        return resp.text.strip()
    print(f"[ClickHouse] Query failed {resp.status_code}: {resp.text[:200]}")
    return None


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _ensure_checkpoint_table(table_name: str) -> None:
    _, _, database, _, _ = _client_config()
    _http_exec(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    _http_exec(f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`{table_name}` (
            environment String,
            checkpoint_value String,
            saved_at DateTime,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, saved_at, checkpoint_value)
    """)


def load_checkpoint_int(table_name: str, environment: str) -> Optional[int]:
    value = load_checkpoint_str(table_name, environment)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_checkpoint_str(table_name: str, environment: str) -> Optional[str]:
    _ensure_checkpoint_table(table_name)
    _, _, database, _, _ = _client_config()
    env_escaped = environment.replace("'", "\\'")
    result = _http_query(
        f"SELECT checkpoint_value FROM `{database}`.`{table_name}` "
        f"WHERE environment = '{env_escaped}' ORDER BY saved_at DESC LIMIT 1"
    )
    return result if result else None


def save_checkpoint_int(table_name: str, environment: str, checkpoint_value: int) -> None:
    save_checkpoint_str(table_name, environment, str(checkpoint_value))


def save_checkpoint_str(table_name: str, environment: str, checkpoint_value: str) -> None:
    _ensure_checkpoint_table(table_name)
    _, _, database, _, _ = _client_config()
    saved_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    env_escaped = environment.replace("'", "\\'")
    val_escaped = str(checkpoint_value).replace("'", "\\'")
    _http_exec(
        f"INSERT INTO `{database}`.`{table_name}` (environment, checkpoint_value, saved_at) "
        f"VALUES ('{env_escaped}', '{val_escaped}', '{saved_at}')"
    )


def ensure_tables(table_names: Iterable[str]) -> None:
    for table_name in table_names:
        _ensure_checkpoint_table(table_name)


# ===========================================================================
# IDP Accuracy – table init, checkpoint, insert
# ===========================================================================

def test_clickhouse_connection(timeout: int = 10) -> bool:
    """Test ClickHouse HTTP connectivity."""
    try:
        host, http_port, database, user, password = _client_config()
        url = f"http://{host}:{http_port}/"
        resp = requests.get(url, auth=(user, password), timeout=timeout, params={"query": "SELECT 1"})
        if resp.status_code == 200:
            logger.info(f"[ClickHouse] Connection successful")
            return True
        logger.error(f"[ClickHouse] Connection failed: {resp.status_code}")
        return False
    except Exception as ex:
        logger.error(f"[ClickHouse] Connection error: {ex}")
        return False


def ensure_database_and_table(timeout: int = 10) -> bool:
    """Ensure database and IDP accuracy transactions table exist."""
    try:
        host, http_port, database, user, password = _client_config()
        url = f"http://{host}:{http_port}/"
        requests.post(url, auth=(user, password), timeout=timeout, data=f"CREATE DATABASE IF NOT EXISTS `{database}`".encode())
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`idp_accuracy_transactions` (
            environment String,
            BatchId String,
            CreatedOn DateTime,
            Filename String,
            ClientCode String,
            PredictedCategory String,
            inserted_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(inserted_at)
        ORDER BY (environment, CreatedOn, BatchId, Filename)
        """
        resp = requests.post(url, auth=(user, password), timeout=timeout, data=ddl.encode())
        if resp.status_code in (200, 201):
            logger.info("[ClickHouse] IDP accuracy table ready")
            return True
        logger.error(f"[ClickHouse] Table creation failed: {resp.status_code} - {resp.text[:200]}")
        return False
    except Exception as ex:
        logger.error(f"[ClickHouse] ensure_database_and_table error: {ex}")
        return False


def insert_idp_transactions_http(
    environment: str,
    records: List[Dict[str, Any]],
    checkpoint_datetime: datetime = None,
    timeout: int = 60
) -> bool:
    """Insert IDP accuracy transaction records via HTTP."""
    if not records:
        return True
    try:
        host, http_port, database, user, password = _client_config()
        url = f"http://{host}:{http_port}/"
        values_list = []
        for idx, record in enumerate(records):
            try:
                created_on = record.get("CreatedOn")
                if isinstance(created_on, str):
                    try:
                        created_on_dt = datetime.fromisoformat(created_on.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        try:
                            created_on_dt = datetime.strptime(created_on[:19], "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            created_on_dt = datetime.utcnow()
                elif isinstance(created_on, datetime):
                    created_on_dt = created_on
                else:
                    created_on_dt = datetime.utcnow()
                created_on_str = created_on_dt.strftime("%Y-%m-%d %H:%M:%S")
                batch_id = str(record.get("BatchId") or "UNKNOWN").replace("'", "\\'")
                filename = str(record.get("Filename") or "UNKNOWN").replace("'", "\\'")
                client_code = str(record.get("ClientCode") or "UNKNOWN").replace("'", "\\'")
                predicted_cat = str(record.get("PredictedCategory") or "UNKNOWN").replace("'", "\\'")
                values_list.append(f"('{environment}', '{batch_id}', '{created_on_str}', '{filename}', '{client_code}', '{predicted_cat}')")
            except Exception as e:
                logger.error(f"[ClickHouse] Error formatting record {idx}: {e}")
                continue
        if not values_list:
            return False
        insert_sql = (
            f"INSERT INTO `{database}`.`idp_accuracy_transactions` "
            f"(environment, BatchId, CreatedOn, Filename, ClientCode, PredictedCategory) "
            f"VALUES {','.join(values_list)}"
        )
        resp = requests.post(url, auth=(user, password), timeout=timeout, data=insert_sql.encode())
        if resp.status_code in (200, 201):
            logger.info(f"[ClickHouse] Inserted {len(values_list)} IDP records")
            return True
        logger.error(f"[ClickHouse] IDP insert failed: {resp.status_code} - {resp.text[:300]}")
        return False
    except Exception as ex:
        logger.error(f"[ClickHouse] insert_idp_transactions_http error: {ex}\n{traceback.format_exc()}")
        return False


def load_idp_accuracy_checkpoint(timeout: int = 10) -> Optional[datetime]:
    """Load last IDP checkpoint from ClickHouse."""
    try:
        host, http_port, database, user, password = _client_config()
        url = f"http://{host}:{http_port}/"
        query = (
            f"SELECT max(inserted_at) FROM `{database}`.`idp_accuracy_transactions` "
            f"WHERE environment = '{get_environment()}'"
        )
        resp = requests.get(url, auth=(user, password), timeout=timeout, params={"query": query})
        if resp.status_code == 200:
            result = resp.text.strip()
            if result and result not in ("0000-00-00 00:00:00", "\\N", ""):
                return datetime.fromisoformat(result)
    except Exception as ex:
        logger.warning(f"[ClickHouse] load_idp_accuracy_checkpoint error: {ex}")
    return None


# ===========================================================================
# Tabak Accuracy – table init, checkpoint, insert
# ===========================================================================

TABAK_SUMMARY_TABLE     = "tabak_accuracy_summary"
TABAK_DAILY_TABLE       = "tabak_accuracy_daily"
TABAK_CATEGORY_TABLE    = "tabak_accuracy_category"
TABAK_SUBCATEGORY_TABLE = "tabak_accuracy_subcategory"


def ensure_tabak_accuracy_tables(timeout: int = 15) -> bool:
    """Create all 4 Tabak accuracy tables if they don't exist."""
    host, http_port, database, user, password = _client_config()
    db = database
    ddls = [
        f"""CREATE TABLE IF NOT EXISTS `{db}`.`{TABAK_SUMMARY_TABLE}` (
            environment String, item_id String, run_datetime DateTime, period String,
            query_start_datetime Nullable(DateTime), query_end_datetime Nullable(DateTime),
            source_date_start Nullable(Date), source_date_end Nullable(Date),
            correct_predictions UInt64, incorrect_predictions UInt64,
            total_records UInt64, accuracy_pct Float64,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, run_datetime, item_id)""",
        f"""CREATE TABLE IF NOT EXISTS `{db}`.`{TABAK_DAILY_TABLE}` (
            environment String, item_id String, run_datetime DateTime, period String,
            source_date Date, correct_predictions UInt64,
            incorrect_predictions UInt64, total_records UInt64, accuracy_pct Float64,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, source_date, run_datetime, item_id)""",
        f"""CREATE TABLE IF NOT EXISTS `{db}`.`{TABAK_CATEGORY_TABLE}` (
            environment String, item_id String, run_datetime DateTime, period String,
            source_date Date, category String, tp UInt64, fp UInt64, fn UInt64,
            precision_pct Float64, recall_pct Float64,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, source_date, category, run_datetime, item_id)""",
        f"""CREATE TABLE IF NOT EXISTS `{db}`.`{TABAK_SUBCATEGORY_TABLE}` (
            environment String, item_id String, run_datetime DateTime, period String,
            source_date Date, subcategory String, tp UInt64, fp UInt64, fn UInt64,
            precision_pct Float64, recall_pct Float64,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, source_date, subcategory, run_datetime, item_id)""",
    ]
    for ddl in ddls:
        if not _http_exec(ddl, timeout=timeout):
            return False
    logger.info("[ClickHouse] Tabak accuracy tables ready")
    return True


def load_tabak_accuracy_checkpoint(environment: str, timeout: int = 10) -> Optional[datetime]:
    """Return max(run_datetime) for the given environment, or None."""
    host, http_port, database, user, password = _client_config()
    env_esc = environment.replace("'", "\\'")
    result = _http_query(
        f"SELECT max(run_datetime) FROM `{database}`.`{TABAK_SUMMARY_TABLE}` WHERE environment = '{env_esc}'",
        timeout=timeout
    )
    if result and result not in ("0000-00-00 00:00:00", "\\N", ""):
        try:
            return datetime.fromisoformat(result)
        except ValueError:
            pass
    return None


def insert_tabak_accuracy(
    environment: str,
    metrics: dict,
    daily_rows: list,
    daily_category_rows: list,
    daily_subcategory_rows: list,
    period: str,
    start_datetime,
    end_datetime,
    run_datetime: datetime,
    timeout: int = 60,
) -> None:
    """Insert Tabak accuracy metrics into all 4 tables via HTTP."""
    host, http_port, database, user, password = _client_config()
    db = database

    def _dt(v) -> str:
        if v is None: return "\\N"
        return v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, "strftime") else str(v)

    def _date(v) -> str:
        if v is None: return "\\N"
        return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]

    def _esc(s) -> str:
        return str(s).replace("'", "\\'")

    run_dt_str = _dt(run_datetime)
    env = _esc(environment)

    _http_exec(
        f"INSERT INTO `{db}`.`{TABAK_SUMMARY_TABLE}` "
        f"(environment,item_id,run_datetime,period,query_start_datetime,query_end_datetime,"
        f"source_date_start,source_date_end,correct_predictions,incorrect_predictions,"
        f"total_records,accuracy_pct) VALUES "
        f"('{env}','summary::{run_dt_str}','{run_dt_str}','{_esc(period)}',"
        f"'{_dt(start_datetime)}','{_dt(end_datetime)}',"
        f"'{_date(metrics.get('source_date_start'))}','{_date(metrics.get('source_date_end'))}',"
        f"{int(metrics['correct'])},{int(metrics['incorrect'])},"
        f"{int(metrics['total'])},{float(metrics['accuracy_pct'])})",
        timeout=timeout,
    )
    if daily_rows:
        parts = [
            f"('{env}','daily::{_date(r['source_date'])}::{run_dt_str}','{run_dt_str}','{_esc(period)}','{_date(r['source_date'])}',"
            f"{int(r['correct_predictions'])},{int(r['incorrect_predictions'])},{int(r['total_records'])},{float(r['accuracy_pct'])})"
            for r in daily_rows
        ]
        _http_exec(f"INSERT INTO `{db}`.`{TABAK_DAILY_TABLE}` (environment,item_id,run_datetime,period,source_date,correct_predictions,incorrect_predictions,total_records,accuracy_pct) VALUES {','.join(parts)}", timeout=timeout)
    if daily_category_rows:
        parts = [
            f"('{env}','category::{_date(r['source_date'])}::{_esc(r['category'])}::{run_dt_str}','{run_dt_str}','{_esc(period)}','{_date(r['source_date'])}','{_esc(r['category'])}',"
            f"{int(r['tp'])},{int(r['fp'])},{int(r['fn'])},{float(r['precision_pct'])},{float(r['recall_pct'])})"
            for r in daily_category_rows
        ]
        _http_exec(f"INSERT INTO `{db}`.`{TABAK_CATEGORY_TABLE}` (environment,item_id,run_datetime,period,source_date,category,tp,fp,fn,precision_pct,recall_pct) VALUES {','.join(parts)}", timeout=timeout)
    if daily_subcategory_rows:
        parts = [
            f"('{env}','subcategory::{_date(r['source_date'])}::{_esc(r['subcategory'])}::{run_dt_str}','{run_dt_str}','{_esc(period)}','{_date(r['source_date'])}','{_esc(r['subcategory'])}',"
            f"{int(r['tp'])},{int(r['fp'])},{int(r['fn'])},{float(r['precision_pct'])},{float(r['recall_pct'])})"
            for r in daily_subcategory_rows
        ]
        _http_exec(f"INSERT INTO `{db}`.`{TABAK_SUBCATEGORY_TABLE}` (environment,item_id,run_datetime,period,source_date,subcategory,tp,fp,fn,precision_pct,recall_pct) VALUES {','.join(parts)}", timeout=timeout)

    logger.info(f"[ClickHouse] Tabak accuracy inserted for run_datetime={run_dt_str}")


# ===========================================================================
# Healthcare EOB / Superbill Accuracy – table init, state query, insert
# ===========================================================================

def _ensure_healthcare_accuracy_table_http(table_name: str, timeout: int = 15) -> None:
    """Create healthcare accuracy table if it doesn't exist."""
    host, http_port, database, user, password = _client_config()
    _http_exec(f"CREATE DATABASE IF NOT EXISTS `{database}`", timeout=timeout)
    _http_exec(f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`{table_name}` (
            environment String,
            item_id String,
            source_type String,
            allocation_id UInt64,
            source_date Nullable(Date),
            date_time Nullable(DateTime),
            file_name String,
            client_name String,
            total_matched UInt64,
            total_mismatches UInt64,
            accuracy Float64,
            created_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (environment, allocation_id, item_id)
    """, timeout=timeout)


def get_healthcare_dataset_state(table_name: str, environment: str) -> tuple:
    """Return (set of item_ids, max allocation_id or None) for the given environment."""
    _ensure_healthcare_accuracy_table_http(table_name)
    host, http_port, database, user, password = _client_config()
    env_esc = environment.replace("'", "\\'")
    result = _http_query(
        f"SELECT item_id, allocation_id FROM `{database}`.`{table_name}` WHERE environment = '{env_esc}' FORMAT TSV"
    )
    item_ids: set = set()
    allocation_ids: list = []
    if result:
        for line in result.splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                iid, aid = parts
                if iid: item_ids.add(iid)
                try: allocation_ids.append(int(aid))
                except ValueError: pass
    return item_ids, (max(allocation_ids) if allocation_ids else None)


def insert_healthcare_accuracy_rows(table_name: str, environment: str, rows: list) -> int:
    """Insert healthcare accuracy rows via HTTP POST."""
    if not rows:
        return 0
    _ensure_healthcare_accuracy_table_http(table_name)
    host, http_port, database, user, password = _client_config()
    url = f"http://{host}:{http_port}/"
    env_esc = environment.replace("'", "\\'")

    def _dt_str(v) -> str:
        if v is None: return "\\N"
        if isinstance(v, (dt.datetime, dt.date)): return v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, dt.datetime) else v.strftime("%Y-%m-%d")
        return str(v)[:19]

    def _date_str(v) -> str:
        if v is None: return "\\N"
        if hasattr(v, "strftime"): return v.strftime("%Y-%m-%d")
        s = str(v)
        return s[:10] if len(s) >= 10 else s

    def _esc(s) -> str:
        return str(s or "").replace("'", "\\'")

    values = []
    for row in rows:
        values.append(
            f"('{env_esc}','{_esc(row['item_id'])}','{_esc(row['source_type'])}',"
            f"{int(row['allocation_id'])},'{_date_str(row.get('date'))}','{_dt_str(row.get('date_time'))}',"
            f"'{_esc(row.get('file_name'))}','{_esc(row.get('client_name'))}',"
            f"{int(row.get('total_matched') or 0)},{int(row.get('total_mismatches') or 0)},{float(row.get('accuracy') or 0.0)})"
        )
    insert_sql = (
        f"INSERT INTO `{database}`.`{table_name}` "
        f"(environment,item_id,source_type,allocation_id,source_date,date_time,"
        f"file_name,client_name,total_matched,total_mismatches,accuracy) "
        f"VALUES {','.join(values)}"
    )
    resp = requests.post(url, auth=(user, password), data=insert_sql.encode(), timeout=60)
    if resp.status_code in (200, 201):
        logger.info(f"[ClickHouse] Inserted {len(rows)} rows into {table_name}")
        return len(rows)
    logger.error(f"[ClickHouse] Healthcare insert failed: {resp.status_code} - {resp.text[:200]}")
    return 0
