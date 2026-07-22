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
#         logging.warning(f"[f2 Tabak FineTuning] Could not read local.settings.json: {e}")
    
#     # Fallback to environment variables for Azure deployment
#     return os.getenv(key, default)

# def process_finetuning_tabak(payload: dict):
#     logging.info("[f2 Tabak FineTuning] Processing payload...")
#     record_ids = payload.get("record_ids", [])
#     environment = payload.get("environment", "exp")
#     file_names = payload.get("File name", [])
#     ground_truth = payload.get("Ground_truth", []) or payload.get("ground_truth", [])

#     if not record_ids:
#         logging.warning("[f2 Tabak FineTuning] No records to process.")
#         return

#     # Load Azure configurations
#     container_name = get_azure_setting("FINAL_DATA_TABAK_CONTAINER", "tabak-dataset")
#     connection_string = get_azure_setting("AzureWebJobsStorage") # standard key for Azure Storage Connection string

#     blob_service_client = None
#     if connection_string:
#         try:
#             blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         except Exception as e:
#             logging.error(f"[f2 Tabak FineTuning] Failed to initialize BlobServiceClient: {e}")
#     else:
#         logging.warning("[f2 Tabak FineTuning] Connection string missing. Azure Blob uploads will be skipped.")

#     tabak_url_prefix = get_azure_setting("TABAK_BLOB_URL_PREFIX", "https://tabakprod.blob.core.windows.net/processed-files/").strip()

#     try:
#         langfuse = Langfuse()
#         dataset_name = f"tabak_fine_tuning_{environment}"
#         langfuse.create_dataset(name=dataset_name, description=f"Tabak Fine Tuning Dataset ({environment})")
#     except Exception as lf_err:
#         logging.error(f"[f2 Tabak FineTuning] Langfuse initialization failed: {lf_err}")
#         return

#     for idx, record_id in enumerate(record_ids):
#         file_name = file_names[idx] if idx < len(file_names) else None
#         gt_item = ground_truth[idx] if idx < len(ground_truth) else None

#         if not file_name or not gt_item:
#             continue

#         category = str(gt_item.get("category") or "Unknown").strip()
#         subcategory = str(gt_item.get("subcategory") or "Default").strip()
        
#         # Sanitize folder path directory segments
#         category_clean = category.replace("/", "_").replace("\\", "_")
#         subcategory_clean = subcategory.replace("/", "_").replace("\\", "_") if subcategory else "Default"
        
#         # Structure pattern: tabak/{category}/{subcategory}/{filename}
#         destination_path = f"tabak/{category_clean}/{subcategory_clean}/{file_name}"

#         # Download PDF from remote blob endpoint and upload to the Tabak finetuning container
#         if blob_service_client and container_name:
#             file_url = f"{tabak_url_prefix.rstrip('/')}/{file_name}"
#             try:
#                 logging.info(f"[f2 Tabak FineTuning] Downloading PDF: {file_url}")
#                 res = requests.get(file_url, timeout=30)
#                 if res.status_code == 200:
#                     pdf_bytes = res.content
                    
#                     blob_client = blob_service_client.get_blob_client(container=container_name, blob=destination_path)
#                     pdf_content_settings = ContentSettings(content_type="application/pdf")
#                     blob_client.upload_blob(
#                         pdf_bytes, 
#                         overwrite=True, 
#                         content_settings=pdf_content_settings
#                     )
#                 else:
#                     logging.error(f"[f2 Tabak FineTuning] Download failed with status: {res.status_code}")
#             except Exception as dl_err:
#                 logging.error(f"[f2 Tabak FineTuning] Error downloading/uploading PDF {file_name}: {dl_err}")

#         # Construct URI for Langfuse
#         storage_account_name = blob_service_client.account_name if blob_service_client else "unknown_account"
#         base_blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"

#         # Synchronize logging item to Langfuse
#         try:
#             langfuse.create_dataset_item(
#                 dataset_name=dataset_name,
#                 id=f"tabak_finetuning::{record_id}",
#                 input={
#                     "record_id": record_id,
#                     "file_name": file_name,
#                     "azure_pdf_path": f"{base_blob_url}/{destination_path}" if blob_service_client else None
#                 },
#                 expected_output=gt_item,
#                 metadata={
#                     "record_type": "tabak_finetuning_record",
#                     "environment": environment
#                 }
#             )
#         except Exception as e:
#             logging.error(f"[f2 Tabak FineTuning] Error writing Langfuse dataset item: {e}")

#     langfuse.flush()
#     logging.info(f"[f2 Tabak FineTuning] tabak fine-tuning batch processed successfully.")



import os
import json
import logging
import requests
import datetime

# Import Azure Blob Storage libraries
from azure.storage.blob import BlobServiceClient, ContentSettings

# Langfuse SDK for fetching pipeline traces/prompts
try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

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


def _parse_ts_from_filename(file_name: str):
    """
    Extracts a datetime from the embedded timestamp in Tabak file names.
    Pattern: ..._{YYYYMMDDHHMMSS}...
    e.g. June_22nd__2026_1-30_ishaan_others__20260622030544392260_0.pdf
    Returns a UTC datetime or None if parsing fails.
    """
    import re
    match = re.search(r'(\d{14})', file_name)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            pass
    return None


def _fetch_prompts_from_tabak_langfuse(file_name: str, langfuse_client) -> list:
    """
    Searches tabak-classificationEngine traces and matches by file name at:
        trace.input["args"][0]["filename"]

    Search window: ±2 hours around the timestamp embedded in the file name
    (format YYYYMMDDHHMMSS), falling back to the last 90 days if parsing fails.

    Prompt path inside each trace:
        tabak-classificationEngine (root SPAN)
            └── GenerateContent (GENERATION) ← prompt at input["contents"][0]["parts"][1]
                                               (there may be 2 GenerateContent per trace)

    Returns a flat list of prompt dicts sorted by observation start_time ascending
    (earliest = _prompt_1.json). Each dict: { trace_id, observation_id, file_name, prompt }
    """
    try:
        file_ts = _parse_ts_from_filename(file_name)
        if file_ts:
            from_ts = file_ts - datetime.timedelta(hours=2)
            to_ts   = file_ts + datetime.timedelta(hours=2)
        else:
            from_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
            to_ts   = None

        matched_traces = []
        MAX_PAGES = 20
        for page in range(1, MAX_PAGES + 1):
            kwargs = dict(name="tabak-classificationEngine", from_timestamp=from_ts, limit=50, page=page)
            if to_ts:
                kwargs["to_timestamp"] = to_ts
            result = langfuse_client.fetch_traces(**kwargs)
            traces = result.data if result and result.data else []
            if not traces:
                break
            for t in traces:
                try:
                    trace_file = (t.input or {}).get("args", [{}])[0].get("filename")
                except Exception:
                    trace_file = None
                if trace_file == file_name:
                    matched_traces.append(t)
            if not result.meta or page >= result.meta.total_pages:
                break

        if not matched_traces:
            logging.warning(f"[f2 Tabak FineTuning] No Langfuse traces found for file: {file_name}")
            return []

        # Collect ALL GenerateContent observations across all matching traces
        all_generate_content = []
        for trace in matched_traces:
            observations = langfuse_client.fetch_observations(trace_id=trace.id)
            obs_list = observations.data if observations and observations.data else []
            for o in obs_list:
                if (o.name or "").strip().lower() == "generatecontent":
                    all_generate_content.append((trace.id, o))

        if not all_generate_content:
            logging.warning(f"[f2 Tabak FineTuning] No GenerateContent observations found for file: {file_name}")
            return []

        # Sort ascending by start_time (earliest = prompt_1)
        all_generate_content.sort(
            key=lambda pair: getattr(pair[1], "start_time", None)
            or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        )

        prompt_list = []
        for trace_id, obs in all_generate_content:
            try:
                prompt_text_part = (obs.input or {})["contents"][0]["parts"][1]
            except (KeyError, IndexError, TypeError):
                prompt_text_part = None
                logging.warning(
                    f"[f2 Tabak FineTuning] Could not extract contents[0].parts[1] "
                    f"from GenerateContent obs '{obs.id}' in trace '{trace_id}'."
                )
            prompt_list.append({
                "trace_id": trace_id,
                "observation_id": obs.id,
                "file_name": file_name,
                "prompt": prompt_text_part,
            })

        return prompt_list
    except Exception as e:
        logging.warning(f"[f2 Tabak FineTuning] Failed to fetch Langfuse traces for '{file_name}': {e}")
        return []


def process_finetuning_tabak(payload: dict):
    logging.info("[f2 Tabak FineTuning] Processing payload...")
    record_ids = payload.get("record_ids", [])
    file_names = payload.get("File name", [])
    ground_truth = payload.get("Ground_truth", []) or payload.get("ground_truth", [])
    source_folder = str(payload.get("folder_name") or payload.get("source") or "tabak").strip()

    if not record_ids:
        logging.warning("[f2 Tabak FineTuning] No records to process.")
        return

    # =================================================================
    # Load Azure configurations for Upload Destination
    # =================================================================
    container_name = get_azure_setting("FINAL_DATA_TABAK_CONTAINER", "tabak-dataset")
    dest_connection_string = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING")

    # =================================================================
    # Load Azure configurations for Fallback Source Download
    # =================================================================
    source_container_name = get_azure_setting("SOURCE_DATA_TABAK_CONTAINER")
    source_connection_string = get_azure_setting("TABAK_STORAGE_CONNECTION_STRING")

    # Initialize Destination Client (For Uploading)
    blob_service_client = None
    if dest_connection_string:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(dest_connection_string)
        except Exception as e:
            logging.error(f"[f2 Tabak FineTuning] Failed to initialize Destination BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Tabak FineTuning] Destination connection string missing. Azure Blob uploads will be skipped.")

    # Initialize Source Client (For Fallback Downloading)
    source_blob_service_client = None
    if source_connection_string:
        try:
            source_blob_service_client = BlobServiceClient.from_connection_string(source_connection_string)
        except Exception as e:
            logging.error(f"[f2 Tabak FineTuning] Failed to initialize Source BlobServiceClient: {e}")
    else:
        logging.warning("[f2 Tabak FineTuning] Source connection string missing. Fallback downloads from blob will fail.")

    # Initialize Langfuse Client (For fetching pipeline traces/prompts)
    langfuse_client = None
    if Langfuse is not None:
        lf_public_key = get_azure_setting("TABAK_LANGFUSE_PUBLIC_KEY")
        lf_secret_key = get_azure_setting("TABAK_LANGFUSE_SECRET_KEY")
        lf_host = get_azure_setting("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if lf_public_key and lf_secret_key:
            try:
                langfuse_client = Langfuse(
                    public_key=lf_public_key,
                    secret_key=lf_secret_key,
                    host=lf_host
                )
                logging.info("[f2 Tabak FineTuning] Langfuse client initialized.")
            except Exception as lf_err:
                logging.warning(f"[f2 Tabak FineTuning] Failed to initialize Langfuse client: {lf_err}")
        else:
            logging.warning("[f2 Tabak FineTuning] Tabak Langfuse keys missing — prompt fetch will be skipped.")
    else:
        logging.warning("[f2 Tabak FineTuning] langfuse package not installed — prompt fetch will be skipped.")

    tabak_url_prefix = get_azure_setting("TABAK_BLOB_URL_PREFIX", "https://tabakprod.blob.core.windows.net/processed-files/").strip()

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
        
        # Structure pattern: {source}/{category}/{subcategory}/{filename}
        destination_path = f"{source_folder}/{category_clean}/{subcategory_clean}/{file_name}"
        
        pdf_bytes = None
        file_url = f"{tabak_url_prefix.rstrip('/')}/{file_name}"

        # =================================================================
        # STEP 1: Primary download from source Azure Container
        # =================================================================
        if source_container_name and source_blob_service_client:
            try:
                logging.info(f"[f2 Tabak FineTuning] Fetching '{file_name}' from source container '{source_container_name}'...")
                source_blob_client = source_blob_service_client.get_blob_client(container=source_container_name, blob=file_name)
                
                # Download blob into memory
                pdf_bytes = source_blob_client.download_blob().readall()
                logging.info(f"[f2 Tabak FineTuning] Successfully fetched PDF from source container.")
            except Exception as src_err:
                logging.warning(f"[f2 Tabak FineTuning] Source blob fetch failed for '{file_name}': {src_err}. Falling back to URL download.")

        # =================================================================
        # STEP 2: FALLBACK - Attempt to download via HTTP URL
        # =================================================================
        if not pdf_bytes:
            try:
                logging.info(f"[f2 Tabak FineTuning] Attempting URL fallback download: {file_url}")
                res = requests.get(file_url, timeout=30)
                if res.status_code == 200:
                    pdf_bytes = res.content
                    logging.info(f"[f2 Tabak FineTuning] Successfully downloaded PDF via URL for record {record_id}.")
                else:
                    logging.warning(f"[f2 Tabak FineTuning] URL fallback download failed with Status {res.status_code}.")
            except Exception as dl_err:
                logging.warning(f"[f2 Tabak FineTuning] URL fallback download error for '{file_name}': {dl_err}")

        # =================================================================
        # STEP 3: Upload the PDF to destination container
        # =================================================================
        if pdf_bytes and blob_service_client and container_name:
            try:
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=destination_path)
                pdf_content_settings = ContentSettings(content_type="application/pdf")
                blob_client.upload_blob(
                    pdf_bytes, 
                    overwrite=True, 
                    content_settings=pdf_content_settings
                )
                logging.info(f"[f2 Tabak FineTuning] Successfully uploaded PDF for record {record_id} to {destination_path}.")
            except Exception as up_err:
                logging.error(f"[f2 Tabak FineTuning] Error uploading PDF for record {record_id}: {up_err}")

            # =============================================================
            # STEP 4: Fetch Langfuse prompt(s) and upload alongside PDF
            # Multiple traces → _prompt_1.json, _prompt_2.json (by time asc)
            # No trace found  → _prompt_1.json with empty {}
            # =============================================================
            if langfuse_client:
                base_name, _ = os.path.splitext(file_name)
                dest_dir = "/".join(destination_path.split("/")[:-1])
                prompts = _fetch_prompts_from_tabak_langfuse(file_name, langfuse_client)

                # Always save at least one file — empty {} if nothing found
                if not prompts:
                    prompts = [{}]

                for i, prompt_data in enumerate(prompts, start=1):
                    prompt_blob_path = f"{dest_dir}/{base_name}_prompt_{i}.json"
                    try:
                        prompt_json_str = json.dumps(prompt_data, indent=2, ensure_ascii=False, default=str)
                        prompt_blob_client = blob_service_client.get_blob_client(
                            container=container_name, blob=prompt_blob_path
                        )
                        prompt_blob_client.upload_blob(
                            prompt_json_str,
                            overwrite=True,
                            content_settings=ContentSettings(content_type="application/json")
                        )
                        logging.info(f"[f2 Tabak FineTuning] Uploaded prompt JSON at '{prompt_blob_path}'.")
                    except Exception as prompt_err:
                        logging.error(f"[f2 Tabak FineTuning] Failed to upload prompt JSON for record {record_id}: {prompt_err}")
        elif not pdf_bytes:
            logging.error(f"[f2 Tabak FineTuning] Could not obtain PDF data for record {record_id}. Skipping.")

    logging.info(f"[f2 Tabak FineTuning] tabak fine-tuning batch processed successfully.")