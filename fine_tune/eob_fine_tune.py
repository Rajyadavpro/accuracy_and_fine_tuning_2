import os
import json
import logging
import requests
from langfuse import Langfuse

# Import Azure Blob Storage libraries
from azure.storage.blob import BlobServiceClient, ContentSettings

def get_azure_setting(key: str, default=None):
    """
    Helper function to load variables from local.settings.json (local dev)
    or from Environment Variables (when deployed to Azure).
    """
    try:
        if os.path.exists("local.settings.json"):
            with open("local.settings.json", "r") as f:
                settings = json.load(f)
                if "Values" in settings and key in settings["Values"]:
                    return settings["Values"][key]
    except Exception as e:
        logging.warning(f"[f2 Healthcare EOB FineTuning] Could not read local.settings.json: {e}")
    
    # Fallback to environment variables for Azure deployment
    return os.getenv(key, default)

def process_finetuning_healthcare_eob(payload: dict):
    logging.info("[f2 Healthcare EOB FineTuning] Processing payload...")
    environment = payload.get("environment", "exp")
    records = payload.get("records", [])

    if not records:
        if "allocation_id" in payload:
            records = [{
                "file_name": payload.get("file_name"),
                "allocation_id": payload.get("allocation_id"),
                "ground_truth": payload.get("ground_truth")
            }]

    if not records:
        logging.warning("[f2 Healthcare EOB FineTuning] No records found to process.")
        return

    # Load Azure configurations
    container_name = get_azure_setting("FINAL_DATA_EOB_CONTAINER", "eob-dataset")
    connection_string = get_azure_setting("AzureWebJobsStorage") # standard key for Azure Storage Connection string

    blob_service_client = None
    if connection_string:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        except Exception as e:
            logging.error(f"[f2 Healthcare EOB FineTuning] Failed to initialize BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Healthcare EOB FineTuning] Connection string missing. Azure Blob uploads will be skipped.")

    try:
        langfuse = Langfuse()
        dataset_name = f"healthcare_eob_fine_tuning_{environment}"
        langfuse.create_dataset(name=dataset_name, description=f"Healthcare EOB Fine Tuning Dataset ({environment})")
    except Exception as lf_err:
        logging.error(f"[f2 Healthcare EOB FineTuning] Langfuse init failed: {lf_err}")
        return

    for rec in records:
        file_name = rec.get("file_name") or "unknown_file.pdf"
        alloc_id = str(rec.get("allocation_id"))
        ground_truth = rec.get("ground_truth") or {}

        if not ground_truth:
            logging.warning(f"[f2 Healthcare EOB FineTuning] Missing ground truth for allocation {alloc_id}.")
            continue

        # Isolate base filename to ensure paired matching files
        base_name, _ = os.path.splitext(file_name)
        pdf_blob_path = f"eob/pdf/{base_name}.pdf"
        json_blob_path = f"eob/json/{base_name}.json"

        # Read SAS URL directly from ground_truth allocation
        file_url = ground_truth.get("Allocation", {}).get("File_url")

        if blob_service_client and container_name:
            # 1. Download and Upload the PDF to Azure Blob Storage
            if file_url:
                try:
                    logging.info(f"[f2 Healthcare EOB FineTuning] Downloading PDF from SAS URL: {file_url[:80]}...")
                    res = requests.get(file_url, timeout=30)
                    if res.status_code == 200:
                        blob_client = blob_service_client.get_blob_client(container=container_name, blob=pdf_blob_path)
                        pdf_content_settings = ContentSettings(content_type="application/pdf")
                        blob_client.upload_blob(
                            res.content, 
                            overwrite=True, 
                            content_settings=pdf_content_settings
                        )
                    else:
                        logging.error(f"[f2 Healthcare EOB FineTuning] Download failed with HTTP Status: {res.status_code}")
                except Exception as dl_err:
                    logging.error(f"[f2 Healthcare EOB FineTuning] Error saving PDF for allocation {alloc_id}: {dl_err}")
            else:
                logging.warning(f"[f2 Healthcare EOB FineTuning] Allocation {alloc_id} lacks 'File_url' path.")

            # 2. Upload the ground truth structure as JSON file to Azure Blob Storage
            try:
                gt_json_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_blob_path)
                json_content_settings = ContentSettings(content_type="application/json")
                blob_client.upload_blob(
                    gt_json_str, 
                    overwrite=True, 
                    content_settings=json_content_settings
                )
            except Exception as json_err:
                logging.error(f"[f2 Healthcare EOB FineTuning] Failed to upload JSON ground truth: {json_err}")

        # Construct URIs for Langfuse (assuming standard Azure Blob URI formatting)
        storage_account_name = blob_service_client.account_name if blob_service_client else "unknown_account"
        base_blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"

        # Update Langfuse Dataset
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"healthcare_eob_finetuning::{alloc_id}",
                input={
                    "allocation_id": alloc_id,
                    "file_name": file_name,
                    "azure_pdf_path": f"{base_blob_url}/{pdf_blob_path}" if blob_service_client else None,
                    "azure_json_path": f"{base_blob_url}/{json_blob_path}" if blob_service_client else None
                },
                expected_output=ground_truth,
                metadata={
                    "record_type": "healthcare_eob_finetuning_record",
                    "environment": environment
                }
            )
        except Exception as e:
            logging.error(f"[f2 Healthcare EOB FineTuning] Error writing dataset item for allocation {alloc_id}: {e}")

    langfuse.flush()
    logging.info(f"[f2 Healthcare EOB FineTuning] EOB fine-tuning batch processing completed.")