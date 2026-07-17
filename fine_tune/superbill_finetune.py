# import os
# import json
# import logging
# import requests
# from langfuse import Langfuse

# # Import Azure Blob Storage libraries
# from azure.storage.blob import BlobServiceClient, ContentSettings

# def get_azure_setting(key: str, default=None):
#     """
#     Helper function to load variables from local.settings.json (local dev)
#     or from Environment Variables (when deployed to Azure).
#     """
#     try:
#         if os.path.exists("local.settings.json"):
#             with open("local.settings.json", "r") as f:
#                 settings = json.load(f)
#                 if "Values" in settings and key in settings["Values"]:
#                     return settings["Values"][key]
#     except Exception as e:
#         logging.warning(f"[f2 Healthcare Superbill FineTuning] Could not read local.settings.json: {e}")
    
#     # Fallback to environment variables for Azure deployment
#     return os.getenv(key, default)

# def process_finetuning_healthcare_superbill(payload: dict):
#     logging.info("[f2 Healthcare Superbill FineTuning] Processing payload...")
#     environment = payload.get("environment", "exp")
#     records = payload.get("records", [])

#     if not records:
#         if "allocation_id" in payload:
#             records = [{
#                 "file_name": payload.get("file_name"),
#                 "allocation_id": payload.get("allocation_id"),
#                 "ground_truth": payload.get("ground_truth")
#             }]

#     if not records:
#         logging.warning("[f2 Healthcare Superbill FineTuning] No records found to process.")
#         return

#     # Load Azure configurations
#     container_name = get_azure_setting("FINAL_DATA_SUPERBILL_CONTAINER", "superbill-dataset")
#     connection_string = get_azure_setting("AzureWebJobsStorage") # standard key for Azure Storage Connection string

#     blob_service_client = None
#     if connection_string:
#         try:
#             blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         except Exception as e:
#             logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to initialize BlobServiceClient: {e}")
#     else:
#         logging.warning("[f2 Healthcare Superbill FineTuning] Connection string missing. Azure Blob uploads will be skipped.")

#     try:
#         langfuse = Langfuse()
#         dataset_name = f"healthcare_superbill_fine_tuning_{environment}"
#         langfuse.create_dataset(name=dataset_name, description=f"Healthcare Superbill Fine Tuning Dataset ({environment})")
#     except Exception as lf_err:
#         logging.error(f"[f2 Healthcare Superbill FineTuning] Langfuse init failed: {lf_err}")
#         return

#     for rec in records:
#         file_name = rec.get("file_name") or "unknown_file.pdf"
#         alloc_id = str(rec.get("allocation_id"))
#         ground_truth = rec.get("ground_truth") or {}

#         if not ground_truth:
#             logging.warning(f"[f2 Healthcare Superbill FineTuning] Missing ground truth for allocation {alloc_id}.")
#             continue

#         # Isolate base filename to ensure paired matching files
#         base_name, _ = os.path.splitext(file_name)
#         pdf_blob_path = f"superbill/pdf/{base_name}.pdf"
#         json_blob_path = f"superbill/json/{base_name}.json"

#         # Read Azure Blob Storage signed SAS URL directly from ground_truth allocation
#         file_url = ground_truth.get("Allocation", {}).get("File_url")

#         if blob_service_client and container_name:
#             # 1. Download and Upload the PDF to Azure Blob Storage
#             if file_url:
#                 try:
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Downloading PDF from SAS URL: {file_url[:80]}...")
#                     res = requests.get(file_url, timeout=30)
#                     if res.status_code == 200:
#                         blob_client = blob_service_client.get_blob_client(container=container_name, blob=pdf_blob_path)
#                         pdf_content_settings = ContentSettings(content_type="application/pdf")
#                         blob_client.upload_blob(
#                             res.content, 
#                             overwrite=True, 
#                             content_settings=pdf_content_settings
#                         )
#                     else:
#                         logging.error(f"[f2 Healthcare Superbill FineTuning] Download failed with HTTP Status: {res.status_code}")
#                 except Exception as dl_err:
#                     logging.error(f"[f2 Healthcare Superbill FineTuning] Error saving PDF for allocation {alloc_id}: {dl_err}")
#             else:
#                 logging.warning(f"[f2 Healthcare Superbill FineTuning] Allocation {alloc_id} lacks 'File_url' path.")

#             # 2. Upload the ground truth structure as JSON file to Azure Blob Storage
#             try:
#                 gt_json_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)
#                 blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_blob_path)
#                 json_content_settings = ContentSettings(content_type="application/json")
#                 blob_client.upload_blob(
#                     gt_json_str, 
#                     overwrite=True, 
#                     content_settings=json_content_settings
#                 )
#             except Exception as json_err:
#                 logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload JSON ground truth: {json_err}")

#         # Construct URIs for Langfuse (assuming standard Azure Blob URI formatting)
#         storage_account_name = blob_service_client.account_name if blob_service_client else "unknown_account"
#         base_blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"

#         # Update Langfuse Dataset
#         try:
#             langfuse.create_dataset_item(
#                 dataset_name=dataset_name,
#                 id=f"healthcare_superbill_finetuning::{alloc_id}",
#                 input={
#                     "allocation_id": alloc_id,
#                     "file_name": file_name,
#                     "azure_pdf_path": f"{base_blob_url}/{pdf_blob_path}" if blob_service_client else None,
#                     "azure_json_path": f"{base_blob_url}/{json_blob_path}" if blob_service_client else None
#                 },
#                 expected_output=ground_truth,
#                 metadata={
#                     "record_type": "healthcare_superbill_finetuning_record",
#                     "environment": environment
#                 }
#             )
#         except Exception as e:
#             logging.error(f"[f2 Healthcare Superbill FineTuning] Error writing dataset item for allocation {alloc_id}: {e}")

#     langfuse.flush()
#     logging.info(f"[f2 Healthcare Superbill FineTuning] Superbill fine-tuning batch processing completed.")



import os
import json
import logging
import requests

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
        logging.warning(f"[f2 Healthcare Superbill FineTuning] Could not read local.settings.json: {e}")
    
    # Fallback to environment variables for Azure deployment
    return os.getenv(key, default)


def process_finetuning_healthcare_superbill(payload: dict):
    logging.info("[f2 Healthcare Superbill FineTuning] Processing payload...")
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
        logging.warning("[f2 Healthcare Superbill FineTuning] No records found to process.")
        return

    # Load Azure configurations for Upload Destination
    container_name = get_azure_setting("FINAL_DATA_SUPERBILL_CONTAINER", "superbill-dataset")
    dest_connection_string = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING") 

    # Load Azure configurations for Fallback Source Download
    source_container_name = get_azure_setting("SOURCE_DATA_SUPERBILL_CONTAINER")
    source_connection_string = get_azure_setting("HEALTHCARE_AI_STORAGE_CONNECTION_STRING")

    # Initialize Destination Client (For Uploading)
    blob_service_client = None
    if dest_connection_string:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(dest_connection_string)
        except Exception as e:
            logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Destination BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Healthcare Superbill FineTuning] Destination connection string missing. Azure Blob uploads will be skipped.")

    # Initialize Source Client (For Fallback Downloading)
    source_blob_service_client = None
    if source_connection_string:
        try:
            source_blob_service_client = BlobServiceClient.from_connection_string(source_connection_string)
        except Exception as e:
            logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Source BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Healthcare Superbill FineTuning] Source connection string missing. Fallback downloads from blob will fail.")


    for rec in records:
        file_name = rec.get("file_name") or "unknown_file.pdf"
        alloc_id = str(rec.get("allocation_id"))
        ground_truth = rec.get("ground_truth") or {}

        if not ground_truth:
            logging.warning(f"[f2 Healthcare Superbill FineTuning] Missing ground truth for allocation {alloc_id}.")
            continue

        # Isolate base filename to use as the folder name
        base_name, _ = os.path.splitext(file_name)
        
        # =================================================================
        # Folder Structure
        # Creates a folder named after the file, with both files inside it
        # =================================================================
        pdf_blob_path = f"{base_name}/{base_name}.pdf"
        json_blob_path = f"{base_name}/{base_name}.json"

        # Read SAS URL directly from ground_truth allocation
        file_url = ground_truth.get("Allocation", {}).get("File_url")
        
        pdf_bytes = None  # Variable to hold the file data in memory

        if blob_service_client and container_name:
            
            # =================================================================
            # STEP 1: Attempt to download via SAS URL
            # =================================================================
            if file_url:
                try:
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Attempting to download PDF from SAS URL...")
                    res = requests.get(file_url, timeout=30)
                    if res.status_code == 200:
                        pdf_bytes = res.content
                        logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully downloaded PDF via URL for allocation {alloc_id}.")
                    else:
                        logging.warning(f"[f2 Healthcare Superbill FineTuning] URL download failed with Status {res.status_code}. Falling back to Blob Storage.")
                except Exception as dl_err:
                    logging.warning(f"[f2 Healthcare Superbill FineTuning] Error downloading PDF from URL: {dl_err}. Falling back to Blob Storage.")
            else:
                logging.warning(f"[f2 Healthcare Superbill FineTuning] Allocation {alloc_id} lacks 'File_url'. Falling back to Blob Storage.")

            # =================================================================
            # STEP 2: FALLBACK - Download from source Azure Container if URL failed
            # =================================================================
            if not pdf_bytes and source_container_name and source_blob_service_client:
                try:
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Fetching '{file_name}' from source container '{source_container_name}'...")
                    source_blob_client = source_blob_service_client.get_blob_client(container=source_container_name, blob=file_name)
                    
                    # Download blob into memory
                    pdf_bytes = source_blob_client.download_blob().readall()
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully fetched PDF from source container.")
                except Exception as src_err:
                    logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to fetch PDF '{file_name}' from source container: {src_err}")

            # =================================================================
            # STEP 3: Upload the PDF to destination container
            # =================================================================
            if pdf_bytes:
                try:
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=pdf_blob_path)
                    pdf_content_settings = ContentSettings(content_type="application/pdf")
                    blob_client.upload_blob(
                        pdf_bytes, 
                        overwrite=True, 
                        content_settings=pdf_content_settings
                    )
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded PDF for allocation {alloc_id}.")
                except Exception as up_err:
                    logging.error(f"[f2 Healthcare Superbill FineTuning] Error uploading PDF for allocation {alloc_id}: {up_err}")
            else:
                logging.error(f"[f2 Healthcare Superbill FineTuning] Could not obtain PDF data for allocation {alloc_id}. Skipping PDF upload.")

            # =================================================================
            # STEP 4: Upload the JSON Ground Truth
            # =================================================================
            try:
                gt_json_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_blob_path)
                json_content_settings = ContentSettings(content_type="application/json")
                blob_client.upload_blob(
                    gt_json_str, 
                    overwrite=True, 
                    content_settings=json_content_settings
                )
                logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded JSON ground truth for allocation {alloc_id}.")
            except Exception as json_err:
                logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload JSON ground truth: {json_err}")

    logging.info(f"[f2 Healthcare Superbill FineTuning] Superbill fine-tuning batch processing completed.")