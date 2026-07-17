import os
import pymssql
import logging
import json
from langfuse import Langfuse

def process_accuracy_idp(payload: dict):
    logging.info("[f2 IDP Accuracy] Processing payload...")
    record_ids = payload.get("record_ids", [])
    environment = payload.get("environment", "exp")
    
    if not record_ids:
        logging.warning("[f2 IDP Accuracy] No record_ids provided.")
        return

    server = os.getenv("IDP_SQL_SERVER")
    database = os.getenv("IDP_SQL_DATABASE")
    user = os.getenv("IDP_SQL_USER")
    password = os.getenv("IDP_SQL_PASSWORD")

    if not all([server, database, user, password]):
        logging.error("[f2 IDP Accuracy] Missing database configurations.")
        return

    host = server.strip()
    port = 1433
    if "," in host:
        host, port_str = host.rsplit(",", 1)
        try:
            port = int(port_str.strip())
        except ValueError:
            pass

    try:
        conn = pymssql.connect(
            server=host,
            port=port,
            user=user,
            password=password,
            database=database,
            login_timeout=30,
            tds_version="7.4"
        )
    except Exception as e:
        logging.error(f"[f2 IDP Accuracy] Database connection failed: {e}", exc_info=True)
        return

    try:
        cursor = conn.cursor(as_dict=True)
        placeholders = ", ".join(["%s"] * len(record_ids))
        query = f"""
            SELECT Id, CreatedOn, ResponsePayload 
            FROM dbo.vw_PdfClassificationTransactionLog 
            WHERE Id IN ({placeholders})
        """
        cursor.execute(query, tuple(record_ids))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"[f2 IDP Accuracy] Query execution failed: {e}", exc_info=True)
        try:
            conn.close()
        except Exception:
            pass
        return

    logging.info(f"[f2 IDP Accuracy] Fetched {len(rows)} matching records from DB.")

    try:
        langfuse = Langfuse()
        dataset_name = f"idp_accuracy_eval_{environment}"
        langfuse.create_dataset(name=dataset_name, description=f"IDP Live Accuracy Evaluations ({environment})")
    except Exception as lf_err:
        logging.error(f"[f2 IDP Accuracy] Langfuse initialization/dataset creation failed: {lf_err}")
        return

    for row in rows:
        record_id = str(row.get("Id"))
        response_payload = row.get("ResponsePayload") or ""
        
        predicted_category = None
        if response_payload:
            try:
                res_json = json.loads(response_payload)
                if isinstance(res_json, dict):
                    predicted_category = res_json.get("Predicted Category") or (
                        res_json.get("json", [{}])[0].get("Predicted Category") 
                        if isinstance(res_json.get("json"), list) and res_json.get("json") 
                        else None
                    )
            except Exception:
                pass

        is_accurate = 0
        if predicted_category and str(predicted_category).strip().lower() == "statement":
            is_accurate = 1

        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"idp_accuracy::{record_id}",
                input={
                    "record_id": record_id,
                    "created_on": str(row.get("CreatedOn")),
                    "predicted_category": predicted_category
                },
                expected_output={
                    "is_accurate": is_accurate
                },
                metadata={
                    "record_type": "idp_accuracy_evaluation",
                    "environment": environment
                }
            )
        except Exception as e:
            logging.error(f"[f2 IDP Accuracy] Failed to create dataset item for ID {record_id}: {e}")

    langfuse.flush()
    logging.info(f"[f2 IDP Accuracy] Processed and synchronized evaluation dataset items to '{dataset_name}'.")