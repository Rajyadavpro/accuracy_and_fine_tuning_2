import azure.functions as func
import logging
import json
import os
from pathlib import Path

app = func.FunctionApp()


def route_fetched_message(payload: dict):
    """
    Decides which worker should execute based on process_type and source.
    """
    process_type = payload.get("process_type")
    
    # Check "file_type" as a fallback if "source" is missing (based on queue payload variations)
    source = payload.get("source") or payload.get("file_type")

    if not process_type or not source:
        logging.warning(f"[f2 Router] Message skipped. Missing process_type or source. Payload: {payload}")
        return

    pt_normalized = str(process_type).strip().lower()
    src_normalized = str(source).strip().lower()

    logging.info(f"[f2 Router] Routing incoming task: Type='{process_type}', Source='{source}'")

    if pt_normalized == "accuracy":
        if src_normalized == "idp":
            from accuracy.idp_accuracy import process_accuracy_idp
            process_accuracy_idp(payload)
        elif src_normalized == "tabak":
            from accuracy.tabak_accuracy import process_accuracy_tabak
            process_accuracy_tabak(payload)
        elif src_normalized in ("healthcare_eob", "healthcare_accuracy_eob", "eob"):
            from accuracy.healthcare_accuracy import process_accuracy_healthcare_eob
            process_accuracy_healthcare_eob(payload)
        elif src_normalized in ("healthcare_superbill", "healthcare_accuracy_superbill", "superbill"):
            from accuracy.healthcare_accuracy import process_accuracy_healthcare_superbill
            process_accuracy_healthcare_superbill(payload)
        else:
            logging.warning(f"[f2 Router] Unhandled Accuracy source: '{source}'")

    elif pt_normalized == "finetuning":
        if src_normalized == "idp":
            from fine_tune.idp_finetune import process_finetuning_idp
            process_finetuning_idp(payload)
        elif src_normalized == "tabak":
            from fine_tune.tabak_finetune import process_finetuning_tabak
            process_finetuning_tabak(payload)
        elif src_normalized in ("healthcare_eob", "eob"):
            from fine_tune.eob_fine_tune import process_finetuning_healthcare_eob
            process_finetuning_healthcare_eob(payload)
        elif src_normalized in ("healthcare_superbill", "superbill"):
            from fine_tune.superbill_finetune import process_finetuning_healthcare_superbill
            process_finetuning_healthcare_superbill(payload)
        elif src_normalized == "ccai":
            from fine_tune.ccai_fine_tune import process_finetuning_ccai
            process_finetuning_ccai(payload)
        else:
            logging.warning(f"[f2 Router] Unhandled FineTuning source: '{source}'")
            
    else:
        logging.warning(f"[f2 Router] Unknown process_type received: '{process_type}'")


# =====================================================================
# SERVICE BUS TRIGGER DEFINITION
# =====================================================================

@app.service_bus_queue_trigger(
    arg_name="msg",
    # Wrapping in % tells Azure Functions to read the value from App Settings / local.settings.json
    queue_name="%SERVICE_BUS_QUEUE_NAME%",
    connection="SERVICE_BUS_CONNECTION_STRING",
    is_sessions_enabled=False,
    auto_complete=False
)
def main_queue_worker_trigger(msg: func.ServiceBusMessage) -> None:
    # If USE_QUEUE toggle is off, skip processing from queue
    if os.getenv("USE_QUEUE", "true").strip().lower() != "true":
        logging.info("f2 Queue Trigger: USE_QUEUE=false — skipping queue message, folder mode is active.")
        return

    logging.info("f2 Queue Trigger activated. Parsing message payload...")
    delete_after_processing = os.getenv("DELETE_MSG_AFTER_PROCESSING", "true").strip().lower() == "true"
    try:
        body_str = msg.get_body().decode('utf-8')
        data = json.loads(body_str)

        # Accommodate wrapped payloads or raw message JSONs
        if "body" in data and isinstance(data["body"], dict):
            payload = data["body"]
        else:
            payload = data

        route_fetched_message(payload)
        if delete_after_processing:
            logging.info("f2 Queue Trigger: Processing succeeded. Runtime will auto-complete message.")
        else:
            logging.info("f2 Queue Trigger: DELETE_MSG_AFTER_PROCESSING=false is not supported with this trigger type; runtime auto-complete remains active.")

    except json.JSONDecodeError:
        logging.error("f2 Trigger failed: Message content was not valid JSON.")
        raise
    except Exception as e:
        logging.error(f"f2 Trigger encountered an execution error: {e}", exc_info=True)
        raise


# =====================================================================
# FOLDER POLLING TRIGGER DEFINITION
# Polls MSG_FOLDER_PATH every 10 seconds for JSON files when USE_QUEUE=false
# =====================================================================

@app.timer_trigger(schedule="*/10 * * * * *", arg_name="timer", run_on_startup=False, use_monitor=False)
def folder_worker_trigger(timer: func.TimerRequest) -> None:
    # Only active when USE_QUEUE toggle is off
    if os.getenv("USE_QUEUE", "true").strip().lower() == "true":
        return

    folder_path = os.getenv("MSG_FOLDER_PATH", "msg_inbox").strip()
    folder = Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"[f2 Folder Trigger] Created inbox folder: {folder.resolve()}")
        return

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return

    processed_folder = folder / "processed"
    failed_folder = folder / "failed"
    processed_folder.mkdir(exist_ok=True)
    failed_folder.mkdir(exist_ok=True)

    for json_file in json_files:
        logging.info(f"[f2 Folder Trigger] Processing file: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Accommodate wrapped payloads or raw message JSONs
            if "body" in data and isinstance(data["body"], dict):
                payload = data["body"]
            else:
                payload = data

            route_fetched_message(payload)

            # Move to processed folder after successful processing
            json_file.rename(processed_folder / json_file.name)
            logging.info(f"[f2 Folder Trigger] Done — moved to processed/: {json_file.name}")

        except json.JSONDecodeError:
            logging.error(f"[f2 Folder Trigger] Invalid JSON in file: {json_file.name}")
            json_file.rename(failed_folder / json_file.name)
        except Exception as e:
            logging.error(f"[f2 Folder Trigger] Failed to process {json_file.name}: {e}", exc_info=True)
            json_file.rename(failed_folder / json_file.name)