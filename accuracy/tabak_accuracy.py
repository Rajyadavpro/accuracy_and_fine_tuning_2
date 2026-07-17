import os
import pymysql
import logging
import json
from langfuse import Langfuse

def clean_text(value):
    return str(value).strip() if value is not None else ""

def canonical_category(value):
    raw = clean_text(value)
    key = raw.lower().replace("_", "").replace(" ", "")
    mapping = {
        "varatingdecision": "VA_Rating_Decision",
        "vafeeletter": "VA_Fee_Letter",
        "other": "Others",
        "others": "Others",
    }
    return mapping.get(key, raw)

def normalize_compare(val1, val2):
    return val1.strip().lower() == val2.strip().lower()

def process_accuracy_tabak(payload: dict):
    logging.info("[f2 Tabak Accuracy] Processing payload...")
    record_ids = payload.get("record_ids", [])
    environment = payload.get("environment", "exp")

    if not record_ids:
        logging.warning("[f2 Tabak Accuracy] No record_ids provided.")
        return

    server = os.getenv("TABAK_DB_SERVER")
    port = os.getenv("TABAK_DB_PORT", "3306")
    database = os.getenv("TABAK_DB_DATABASE")
    user = os.getenv("TABAK_DB_USERID")
    password = os.getenv("TABAK_DB_PASSWORD")

    if not all([server, database, user, password]):
        logging.error("[f2 Tabak Accuracy] Missing database credentials.")
        return

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
        logging.error(f"[f2 Tabak Accuracy] Database connection failed: {e}", exc_info=True)
        return

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
        logging.error(f"[f2 Tabak Accuracy] Query execution failed: {e}", exc_info=True)
        return

    logging.info(f"[f2 Tabak Accuracy] Fetched {len(rows)} matching transaction records.")

    try:
        langfuse = Langfuse()
        dataset_name = f"tabak_accuracy_eval_{environment}"
        langfuse.create_dataset(name=dataset_name, description=f"Tabak live accuracy evaluations ({environment})")
    except Exception as lf_err:
        logging.error(f"[f2 Tabak Accuracy] Langfuse initialization failed: {lf_err}")
        return

    for row in rows:
        t_id = str(row.get("Transation_id"))
        template_info_str = row.get("template_info") or ""
        if not template_info_str:
            continue

        try:
            template_info = json.loads(template_info_str)
        except Exception:
            continue

        generated_response = template_info.get("generated_response", {}).get("VADetails", {})
        user_selected_response = template_info.get("user_selected_response", {}).get("VADetails", {})

        gen_cat = canonical_category(generated_response.get("Category"))
        gen_sub = clean_text(generated_response.get("Subcategory"))
        user_cat = canonical_category(user_selected_response.get("Category"))
        user_sub = clean_text(user_selected_response.get("Subcategory"))

        actual_cat = user_cat if user_cat else gen_cat
        actual_sub = user_sub if user_sub else gen_sub

        is_correct = False
        if (not user_cat and not user_sub) or (normalize_compare(user_cat, gen_cat) and normalize_compare(user_sub, gen_sub)):
            is_correct = True

        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"tabak_accuracy::{t_id}",
                input={
                    "record_id": t_id,
                    "generated_category": gen_cat,
                    "generated_subcategory": gen_sub,
                },
                expected_output={
                    "actual_category": actual_cat,
                    "actual_subcategory": actual_sub,
                    "is_correct": is_correct
                },
                metadata={
                    "record_type": "tabak_accuracy_evaluation",
                    "environment": environment
                }
            )
        except Exception as e:
            logging.error(f"[f2 Tabak Accuracy] Failed to write item for ID {t_id}: {e}")

    langfuse.flush()
    logging.info(f"[f2 Tabak Accuracy] Synchronized evaluation records to Langfuse dataset '{dataset_name}'.")