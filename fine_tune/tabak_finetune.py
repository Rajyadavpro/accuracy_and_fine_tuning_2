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
        logging.warning(f"[f2 Tabak FineTuning] Could not read local.settings.json: {e}")
    
    # Fallback to environment variables for Azure deployment
    return os.getenv(key, default)

def process_finetuning_tabak(payload: dict):
    logging.info("[f2 Tabak FineTuning] Processing payload...")
    record_ids = payload.get("record_ids", [])
    environment = payload.get("environment", "exp")
    file_names = payload.get("File name", [])
    ground_truth = payload.get("Ground_truth", []) or payload.get("ground_truth", [])

    if not record_ids:
        logging.warning("[f2 Tabak FineTuning] No records to process.")
        return

    # Load Azure configurations
    container_name = get_azure_setting("FINAL_DATA_TABAK_CONTAINER", "tabak-dataset")
    connection_string = get_azure_setting("AzureWebJobsStorage") # standard key for Azure Storage Connection string

    blob_service_client = None
    if connection_string:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        except Exception as e:
            logging.error(f"[f2 Tabak FineTuning] Failed to initialize BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Tabak FineTuning] Connection string missing. Azure Blob uploads will be skipped.")

    tabak_url_prefix = get_azure_setting("TABAK_BLOB_URL_PREFIX", "https://tabakprod.blob.core.windows.net/processed-files/").strip()

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

        # Download PDF from remote blob endpoint and upload to the Tabak finetuning container
        if blob_service_client and container_name:
            file_url = f"{tabak_url_prefix.rstrip('/')}/{file_name}"
            try:
                logging.info(f"[f2 Tabak FineTuning] Downloading PDF: {file_url}")
                res = requests.get(file_url, timeout=30)
                if res.status_code == 200:
                    pdf_bytes = res.content
                    
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=destination_path)
                    pdf_content_settings = ContentSettings(content_type="application/pdf")
                    blob_client.upload_blob(
                        pdf_bytes, 
                        overwrite=True, 
                        content_settings=pdf_content_settings
                    )
                else:
                    logging.error(f"[f2 Tabak FineTuning] Download failed with status: {res.status_code}")
            except Exception as dl_err:
                logging.error(f"[f2 Tabak FineTuning] Error downloading/uploading PDF {file_name}: {dl_err}")

        # Construct URI for Langfuse
        storage_account_name = blob_service_client.account_name if blob_service_client else "unknown_account"
        base_blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"

        # Synchronize logging item to Langfuse
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=f"tabak_finetuning::{record_id}",
                input={
                    "record_id": record_id,
                    "file_name": file_name,
                    "azure_pdf_path": f"{base_blob_url}/{destination_path}" if blob_service_client else None
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