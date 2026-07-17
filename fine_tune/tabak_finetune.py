import os
import logging
import requests
from langfuse import Langfuse
from fine_tune.gcp_helper import upload_bytes_to_gcs

def process_finetuning_tabak(payload: dict):
    logging.info("[f2 Tabak FineTuning] Processing payload...")
    record_ids = payload.get("record_ids", [])
    environment = payload.get("environment", "exp")
    file_names = payload.get("File name", [])
    ground_truth = payload.get("Ground_truth", []) or payload.get("ground_truth", [])

    if not record_ids:
        logging.warning("[f2 Tabak FineTuning] No records to process.")
        return

    bucket_name = os.getenv("GCP_FINE_TUNING_BUCKET_NAME")
    if not bucket_name:
        logging.warning("[f2 Tabak FineTuning] GCP_FINE_TUNING_BUCKET_NAME not set. GCS uploads skipped.")

    tabak_url_prefix = os.getenv("TABAK_BLOB_URL_PREFIX", "https://tabakprod.blob.core.windows.net/processed-files/").strip()

    try:
        langfuse = Langfuse()
        dataset_name = f"tabak_fine_tuning_{environment}"
        langfuse.create_dataset(name=dataset_name, description=f"Tabak Fine Tuning Dataset ({environment})")
    except Exception as lf_err:
        logging.error(f"[f2 Tabak FineTuning] Langfuse initialization failed: {lf_err}")
        return

    for idx, record_id in enumerate(record_ids):
        file_name = file_names[idx] if idx < len(file_names) else None
        gt_item = ground_truth[idx] if idx < len(ground_truth) else None

        if not file_name or not gt_item:
            continue

        category = str(gt_item.get("category") or "Unknown").strip()
        subcategory = str(gt_item.get("subcategory") or "Default").strip()
        
        # Sanitize folder path directory segments
        category_clean = category.replace("/", "_").replace("\\", "_")
        subcategory_clean = subcategory.replace("/", "_").replace("\\", "_") if subcategory else "Default"
        
        # Structure pattern: tabak/{category}/{subcategory}/{filename}
        destination_path = f"tabak/{category_clean}/{subcategory_clean}/{file_name}"

        # Download PDF from remote blob endpoint
        if bucket_name:
            file_url = f"{tabak_url_prefix.rstrip('/')}/{file_name}"
            try:
                logging.info(f"[f2 Tabak FineTuning] Downloading PDF: {file_url}")
                res = requests.get(file_url, timeout=30)
                if res.status_code == 200:
                    pdf_bytes = res.content
                    upload_bytes_to_gcs(
                        bucket_name=bucket_name,
                        destination_blob_name=destination_path,
                        data=pdf_bytes,
                        content_type="application/pdf"
                    )
                else:
                    logging.error(f"[f2 Tabak FineTuning] Download failed with status: {res.status_code}")
            except Exception as dl_err:
                logging.error(f"[f2 Tabak FineTuning] Error downloading/uploading PDF {file_name}: {dl_err}")

        # Synchronize logging item to Langfuse
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"tabak_finetuning::{record_id}",
                input={
                    "record_id": record_id,
                    "file_name": file_name,
                    "gcs_path": f"gs://{bucket_name}/{destination_path}" if bucket_name else None
                },
                expected_output=gt_item,
                metadata={
                    "record_type": "tabak_finetuning_record",
                    "environment": environment
                }
            )
        except Exception as e:
            logging.error(f"[f2 Tabak FineTuning] Error writing Langfuse dataset item: {e}")

    langfuse.flush()
    logging.info(f"[f2 Tabak FineTuning] tabak fine-tuning batch processed successfully.")