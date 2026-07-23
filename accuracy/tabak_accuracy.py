import os
import json
import logging
import datetime
import pymysql
import requests
from typing import List, Dict, Any
from langfuse import Langfuse

# Import ClickHouse helpers from your existing module
from clickhouse_store import (
    _client_config,
    _http_exec,
    get_environment,
    ensure_tabak_transaction_metrics_table
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")


# ==========================================
# Helper Functions
# ==========================================

def clean_text(value) -> str:
    """Cleans and standardizes text input[cite: 2]."""
    return str(value).strip() if value is not None else ""


def canonical_category(value) -> str:
    """Maps varied category spelling structures to standard class designations[cite: 2, 4]."""
    raw = clean_text(value)
    key = raw.lower().replace("_", "").replace(" ", "")
    mapping = {
        "varatingdecision": "VA_Rating_Decision",
        "vafeeletter": "VA_Fee_Letter",
        "other": "Others",
        "others": "Others",
    }
    return mapping.get(key, raw)


def normalize_compare(val1: str, val2: str) -> bool:
    """Performs case-insensitive and whitespace-insensitive string matching[cite: 2, 4]."""
    return val1.strip().lower() == val2.strip().lower()


def get_azure_setting(key: str, default=None):
    """Loads variables from local.settings.json or environment variables[cite: 3]."""
    try:
        if os.path.exists("local.settings.json"):
            with open("local.settings.json", "r") as f:
                settings = json.load(f)
                if "Values" in settings and key in settings["Values"]:
                    return settings["Values"][key]
    except Exception as e:
        logging.warning(f"Could not read local.settings.json: {e}")
    return os.getenv(key, default)


# ==========================================
# Metrics & ClickHouse Handlers
# ==========================================

def _parse_date_from_filename(file_name: str) -> datetime.date | None:
    """Extract YYYYMMDD date embedded in filenames like *_20260619112812_*.pdf"""
    import re
    m = re.search(r"(\d{8})\d{6}", file_name)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    return None


def fetch_langfuse_metrics(file_name: str, langfuse_client: Langfuse) -> Dict[str, Any]:
    """
    Searches tabak-classificationEngine traces matching the file name in Langfuse
    to retrieve token counts, total cost, and latency.
    Narrows the search window using the date embedded in the filename.
    Token counts come from child observations since trace.usage is None.
    """
    metrics = {
        "input_token": 0,
        "output_token": 0,
        "cost": 0.0,
        "latency": 0.0
    }

    if not langfuse_client or not file_name:
        return metrics

    try:
        # Narrow time window using date in filename; fallback to last 90 days
        file_date = _parse_date_from_filename(file_name)
        if file_date:
            from_ts = datetime.datetime(file_date.year, file_date.month, file_date.day,
                                        tzinfo=datetime.timezone.utc)
            to_ts = from_ts + datetime.timedelta(days=2)
        else:
            from_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
            to_ts = None

        fetch_kwargs = dict(name="tabak-classificationEngine", from_timestamp=from_ts, limit=100)
        if to_ts:
            fetch_kwargs["to_timestamp"] = to_ts

        result = langfuse_client.fetch_traces(**fetch_kwargs)
        traces = result.data if result and result.data else []

        matched_trace_id = None
        for t in traces:
            inp = t.input or {}
            try:
                args = inp.get("args") or []
                trace_file = (
                    inp.get("filename")
                    or inp.get("file_name")
                    or (args[0].get("filename") if args else None)
                    or (args[0].get("file_name") if args else None)
                )
            except Exception:
                trace_file = None

            if trace_file and trace_file == file_name:
                metrics["cost"] = float(getattr(t, "total_cost", 0.0) or 0.0)
                metrics["latency"] = float(getattr(t, "latency", 0.0) or 0.0)
                matched_trace_id = t.id
                break

        # Trace.usage is None — get tokens from child observations (generations)
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
        logging.warning(f"Failed to fetch Langfuse metrics for '{file_name}': {e}")

    return metrics


def insert_tabak_transaction_metrics(environment: str, rows: List[Dict[str, Any]], timeout: int = 60) -> bool:
    """Inserts processed transaction records into the single ClickHouse table[cite: 5]."""
    if not rows:
        return True

    # Ensure table exists before inserting[cite: 5]
    if not ensure_tabak_transaction_metrics_table(timeout=timeout):
        return False

    host, http_port, database, user, password = _client_config()
    
    def _esc(s: str) -> str:
        return str(s or "").replace("\\", "\\\\").replace("'", "\\'")

    values = []
    for row in rows:
        values.append(
            f"('{_esc(environment)}', '{_esc(row['transaction_id'])}', '{_esc(row['file_name'])}', "
            f"'{_esc(row['gen_cat'])}', '{_esc(row['gen_subcat'])}', '{_esc(row['user_cat'])}', "
            f"'{_esc(row['user_sub_cat'])}', '{_esc(row['category'])}', '{_esc(row['sub_category'])}', "
            f"{int(row['is_cat_correct'])}, {int(row['is_sub_cstcorrect'])}, "
            f"{int(row['input_token'])}, {int(row['output_token'])}, "
            f"{float(row['cost'])}, {float(row['latency'])})"
        )

    insert_sql = f"""
        INSERT INTO `{database}`.`tabak_transaction_metrics` (
            environment, transaction_id, file_name,
            gen_cat, gen_subcat, user_cat, user_sub_cat,
            category, sub_category, is_cat_correct, is_sub_cstcorrect,
            input_token, output_token, cost, latency
        ) VALUES {','.join(values)}
    """

    url = f"http://{host}:{http_port}/"
    resp = requests.post(url, auth=(user, password), data=insert_sql.encode('utf-8'), timeout=timeout)
    
    if resp.status_code in (200, 201):
        logging.info(f"[ClickHouse] Successfully inserted {len(rows)} records into 'tabak_transaction_metrics'.")
        return True
    else:
        logging.error(f"[ClickHouse] Insertion failed ({resp.status_code}): {resp.text[:300]}")
        return False


# ==========================================
# Main Orchestrator
# ==========================================

def process_tabak_transaction_metrics(payload: dict):
    """
    Main function to process Tabak transaction IDs, parse accuracy metrics,
    query Langfuse, and store results directly into ClickHouse[cite: 2, 3, 5].
    """
    logging.info("[f2 Tabak] Processing payload...")
    
    # Safely get IDs using either the new 'id' array format or the legacy 'record_ids' format[cite: 1, 2]
    record_ids = payload.get("id") or payload.get("record_ids", [])
    environment = payload.get("environment", "exp")

    if not record_ids:
        logging.warning("[f2 Tabak] No transaction IDs provided.")
        return

    # Load MariaDB credentials — reads local.settings.json first, then env vars
    server = get_azure_setting("TABAK_DB_SERVER")
    port = get_azure_setting("TABAK_DB_PORT", "3306")
    database = get_azure_setting("TABAK_DB_DATABASE")
    user = get_azure_setting("TABAK_DB_USERID")
    password = get_azure_setting("TABAK_DB_PASSWORD")

    if not all([server, database, user, password]):
        logging.error("[f2 Tabak] Missing MariaDB database credentials. Check environment variables.")
        return

    # Connect to MariaDB[cite: 2]
    try:
        conn = pymysql.connect(
            host=server,
            port=int(port),
            user=user,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
            charset="utf8mb4"
        )
    except Exception as e:
        logging.error(f"[f2 Tabak] Database connection failed: {e}", exc_info=True)
        return

    # Fetch rows from MariaDB[cite: 2]
    try:
        with conn:
            with conn.cursor() as cursor:
                placeholders = ", ".join(["%s"] * len(record_ids))
                query = f"""
                    SELECT Transation_id, template_info 
                    FROM `Transactions` 
                    WHERE Transation_id IN ({placeholders})
                """
                cursor.execute(query, tuple(record_ids))
                rows = cursor.fetchall()
    except Exception as e:
        logging.error(f"[f2 Tabak] DB query execution failed: {e}", exc_info=True)
        return

    logging.info(f"[f2 Tabak] Fetched {len(rows)} matching transaction records from DB[cite: 2].")

    # Initialize Langfuse client[cite: 3]
    langfuse_client = None
    lf_public_key = get_azure_setting("TABAK_LANGFUSE_PUBLIC_KEY")
    lf_secret_key = get_azure_setting("TABAK_LANGFUSE_SECRET_KEY")
    lf_host = get_azure_setting("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if lf_public_key and lf_secret_key:
        try:
            langfuse_client = Langfuse(public_key=lf_public_key, secret_key=lf_secret_key, host=lf_host)
            logging.info("[f2 Tabak] Langfuse client initialized[cite: 3].")
        except Exception as lf_err:
            logging.warning(f"[f2 Tabak] Langfuse initialization failed: {lf_err}[cite: 3].")

    clickhouse_rows = []

    # Process each record
    for row in rows:
        t_id = str(row.get("Transation_id"))
        template_info_str = row.get("template_info") or ""
        
        if not template_info_str:
            continue

        try:
            template_info = json.loads(template_info_str)
        except Exception:
            continue

        # Target the inner VADetails dictionaries[cite: 2]
        generated_response = template_info.get("generated_response", {}).get("VADetails", {})
        user_selected_response = template_info.get("user_selected_response", {}).get("VADetails", {})

        # Extract file_name — try multiple locations in the payload
        def _extract_file_name(ti: dict, gen: dict) -> str:
            for fp in [
                ti.get("filename"),
                ti.get("file_name"),
                ti.get("file_path"),
                gen.get("filename"),
                gen.get("file_name"),
                gen.get("file_path"),
            ]:
                if fp:
                    # handle both / and \ separators
                    return fp.replace("\\", "/").split("/")[-1]
            return ""
        file_name = _extract_file_name(template_info, generated_response)

        # Parse generated and user-selected categories[cite: 2]
        gen_cat = canonical_category(generated_response.get("Category"))
        gen_subcat = clean_text(generated_response.get("Subcategory"))
        user_cat = canonical_category(user_selected_response.get("Category"))
        user_sub_cat = clean_text(user_selected_response.get("Subcategory"))

        # Final category resolution[cite: 2]
        category = user_cat if user_cat else gen_cat
        sub_category = user_sub_cat if user_sub_cat else gen_subcat

        # Category and subcategory correctness flags[cite: 2, 4]
        is_cat_correct = 1 if (not user_cat or normalize_compare(user_cat, gen_cat)) else 0
        is_sub_cstcorrect = 1 if (not user_sub_cat or normalize_compare(user_sub_cat, gen_subcat)) else 0

        # Fetch traces/metrics from Langfuse using the extracted file name[cite: 3]
        lf_metrics = fetch_langfuse_metrics(file_name, langfuse_client)

        clickhouse_rows.append({
            "transaction_id": t_id,
            "file_name": file_name,
            "gen_cat": gen_cat,
            "gen_subcat": gen_subcat,
            "user_cat": user_cat,
            "user_sub_cat": user_sub_cat,
            "category": category,
            "sub_category": sub_category,
            "is_cat_correct": is_cat_correct,
            "is_sub_cstcorrect": is_sub_cstcorrect,
            "input_token": lf_metrics["input_token"],
            "output_token": lf_metrics["output_token"],
            "cost": lf_metrics["cost"],
            "latency": lf_metrics["latency"]
        })

    # Push to ClickHouse[cite: 5]
    if clickhouse_rows:
        insert_tabak_transaction_metrics(environment, clickhouse_rows)


# ==========================================
# Local Testing Entry Point
# ==========================================
if __name__ == "__main__":
    # Test payload utilizing the msg_2208
    
    process_tabak_transaction_metrics()