import os
import logging
import json
import requests
from langfuse import Langfuse
from fine_tune.gcp_helper import upload_bytes_to_gcs

def process_finetuning_idp(payload: dict):
    logging.info("[f2 IDP FineTuning] Processing payload...")
    environment = payload.get("environment", "exp")
    folder_name = str(payload.get("folder_name") or "").strip()

    # Configuration for GCS Bucket
    bucket_name = os.getenv("GCP_FINE_TUNING_BUCKET_NAME")
    if not bucket_name:
        logging.warning("[f2 IDP FineTuning] GCP_FINE_TUNING_BUCKET_NAME not set. GCS uploads skipped.")

    idp_url_prefix = os.getenv("IDP_BLOB_URL_PREFIX", "https://idpprod.blob.core.windows.net/processed-files/").strip()

    # Normalized records list built dynamically based on queue payload contents
    processed_records = []

    # Case A: Queue message contains pre-parsed "File name" and "Ground_truth" (Tabak style)
    if "File name" in payload and ("Ground_truth" in payload or "ground_truth" in payload):
        record_ids = payload.get("record_ids", [])
        file_names = payload.get("File name", [])
        ground_truth = payload.get("Ground_truth", []) or payload.get("ground_truth", [])
        
        for idx, record_id in enumerate(record_ids):
            filename = file_names[idx] if idx < len(file_names) else None
            gt_item = ground_truth[idx] if idx < len(ground_truth) else None
            if not filename:
                continue
                
            category = "Unknown"
            if isinstance(gt_item, dict):
                category = gt_item.get("category") or gt_item.get("Predicted Category") or "Unknown"
            elif isinstance(gt_item, str):
                category = gt_item
                
            processed_records.append({
                "record_id": str(record_id),
                "filename": filename,
                "prediction": category,
                "client_name": "Unknown",
                "ground_truth": gt_item
            })

    # Case B: Queue message contains nested "records" array (Healthcare style)
    elif "records" in payload:
        for rec in payload.get("records", []):
            filename = rec.get("file_name") or rec.get("filename")
            record_id = rec.get("allocation_id") or rec.get("record_id")
            gt_item = rec.get("ground_truth")
            
            if not filename or not record_id:
                continue
                
            category = "Unknown"
            if isinstance(gt_item, dict):
                category = gt_item.get("category") or gt_item.get("Predicted Category") or "Unknown"
                
            processed_records.append({
                "record_id": str(record_id),
                "filename": filename,
                "prediction": category,
                "client_name": "Unknown",
                "ground_truth": gt_item
            })

    # Case C: Queue message forwards raw "ResponsePayload" strings/dicts directly from f1
    else:
        record_ids = payload.get("record_ids", [])
        payloads = payload.get("ResponsePayload") or payload.get("response_payload") or payload.get("payloads") or []
        
        if not isinstance(payloads, list):
            payloads = [payloads]
            
        for idx, record_id in enumerate(record_ids):
            payload_str = payloads[idx] if idx < len(payloads) else None
            if not payload_str:
                continue
                
            try:
                if isinstance(payload_str, str):
                    payload_data = json.loads(payload_str)
                else:
                    payload_data = payload_str
            except Exception:
                continue
                
            prediction = None
            filename = None
            client_name = None

            # Read client identifier from ResponsePayload root first
            if isinstance(payload_data, dict):
                client_name = payload_data.get("client_code") or payload_data.get("clientCode")

            # Targeted matching rules translated from your original script
            if isinstance(payload_data, dict) and isinstance(payload_data.get("json"), list) and payload_data["json"]:
                first_item = payload_data["json"][0]
                if isinstance(first_item, dict):
                    filename = first_item.get("File Name")
                    prediction = first_item.get("Predicted Category")
                    client_name = client_name or (
                        first_item.get("Client Code")
                        or first_item.get("Client Name")
                        or first_item.get("Client")
                        or first_item.get("ClientName")
                        or first_item.get("client_name")
                        or first_item.get("client_code")
                        or first_item.get("clientCode")
                    )

            # Fallback values
            filename = filename or f"transaction_{record_id}.pdf"
            prediction = prediction or "Unknown"
            client_name = client_name or "Unknown"
            
            processed_records.append({
                "record_id": str(record_id),
                "filename": filename,
                "prediction": prediction,
                "client_name": client_name,
                "ground_truth": {
                    "category": prediction,
                    "client_name": client_name
                }
            })

    if not processed_records:
        logging.warning("[f2 IDP FineTuning] No matching structures could be resolved from queue message.")
        return

    # Initialize Langfuse
    try:
        langfuse = Langfuse()
        dataset_name = f"idp_fine_tuning_{environment}"
        langfuse.create_dataset(name=dataset_name, description=f"IDP Fine Tuning Dataset ({environment})")
    except Exception as lf_err:
        logging.error(f"[f2 IDP FineTuning] Langfuse initialization failed: {lf_err}")
        return

    # Process and upload individual elements
    for rec in processed_records:
        record_id = rec["record_id"]
        filename = rec["filename"]
        prediction = rec["prediction"]
        client_name = rec["client_name"]
        gt_item = rec["ground_truth"]

        # Format classification path: {folder_name}/{prediction}/{filename}
        prediction_clean = str(prediction).strip().replace("/", "_").replace("\\", "_")
        _prefix = f"{folder_name}/" if folder_name else ""
        destination_path = f"{_prefix}{prediction_clean}/{filename}"

        # Sync PDF file binaries directly to GCS
        if bucket_name:
            # Skip download for faked backup filenames
            if not filename.startswith("transaction_"):
                file_url = f"{idp_url_prefix.rstrip('/')}/{filename}"
                try:
                    logging.info(f"[f2 IDP FineTuning] Downloading PDF: {file_url}")
                    res = requests.get(file_url, timeout=30)
                    if res.status_code == 200:
                        upload_bytes_to_gcs(
                            bucket_name=bucket_name,
                            destination_blob_name=destination_path,
                            data=res.content,
                            content_type="application/pdf"
                        )
                    else:
                        logging.error(f"[f2 IDP FineTuning] Download failed for {filename} (HTTP Status {res.status_code})")
                except Exception as upload_err:
                    logging.error(f"[f2 IDP FineTuning] Error handling file GCS sync for {filename}: {upload_err}")

        # Sync record parameters to Langfuse Dataset
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"idp_finetuning::{record_id}",
                input={
                    "record_id": record_id,
                    "file_name": filename,
                    "client_name": client_name,
                    "gcs_path": f"gs://{bucket_name}/{destination_path}" if bucket_name else None
                },
                expected_output=gt_item,
                metadata={
                    "record_type": "idp_finetuning_record",
                    "environment": environment
                }
            )
        except Exception as lf_write_err:
            logging.error(f"[f2 IDP FineTuning] Failed to write dataset item for transaction {record_id}: {lf_write_err}")

    langfuse.flush()
    logging.info(f"[f2 IDP FineTuning] Uploaded {len(processed_records)} items to Langfuse dataset '{dataset_name}'.")