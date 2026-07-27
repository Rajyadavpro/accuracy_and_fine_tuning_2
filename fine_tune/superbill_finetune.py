
# import os
# import json
# import logging
# import requests

# # Import Azure Blob Storage libraries
# from azure.storage.blob import BlobServiceClient, ContentSettings

# # Langfuse SDK for fetching pipeline traces/prompts
# try:
#     from langfuse import Langfuse
# except ImportError:
#     Langfuse = None

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


# def _fetch_prompt_from_langfuse(file_name: str, langfuse_client) -> dict | None:
#     """
#     Searches Medical-Billing-Pipeline traces in Langfuse, matches by
#     metadata["file_name"], then navigates:
#       - Gemini Extraction (GENERATION) → GenerateContent (GENERATION) → input
#       - OR CMS Form Extraction (GENERATION) → input directly
#     Searches only the last 90 days, capped at 5 pages (250 traces).
#     Returns a dict with trace_id, trace_name, file_name, and prompt input,
#     or None if the trace/observation is not found.
#     """
#     import datetime
#     try:
#         from_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)

#         matched_trace = None
#         MAX_PAGES = 5
#         for page in range(1, MAX_PAGES + 1):
#             result = langfuse_client.fetch_traces(
#                 name="Medical-Billing-Pipeline",
#                 from_timestamp=from_ts,
#                 limit=50,
#                 page=page
#             )
#             traces = result.data if result and result.data else []
#             if not traces:
#                 break
#             for t in traces:
#                 meta = t.metadata or {}
#                 if meta.get("file_name") == file_name:
#                     matched_trace = t
#                     break
#             if matched_trace:
#                 break
#             if not result.meta or page >= result.meta.total_pages:
#                 break

#         if not matched_trace:
#             logging.warning(f"[f2 Superbill FineTuning] No Langfuse trace found for file: {file_name}")
#             return None

#         # Fetch all observations for the matched trace
#         observations = langfuse_client.fetch_observations(trace_id=matched_trace.id)
#         obs_list = observations.data if observations and observations.data else []

#         # Build parent → [children] map
#         children_of: dict = {}
#         for o in obs_list:
#             parent = getattr(o, "parent_observation_id", None)
#             children_of.setdefault(parent, []).append(o)

#         prompt_input = None

#         # Path 1: Gemini Extraction → GenerateContent → input
#         gemini_extraction = next(
#             (o for o in obs_list if (o.name or "").strip().lower() == "gemini extraction"),
#             None
#         )
#         if gemini_extraction:
#             generate_content = next(
#                 (o for o in children_of.get(gemini_extraction.id, [])
#                  if (o.name or "").strip().lower() == "generatecontent"),
#                 None
#             )
#             if generate_content:
#                 prompt_input = generate_content.input
#             else:
#                 logging.warning(
#                     f"[f2 Superbill FineTuning] 'GenerateContent' not found inside "
#                     f"'Gemini Extraction' for trace '{matched_trace.id}'."
#                 )

#         # Path 2: CMS Form Extraction (GENERATION) → input directly
#         if prompt_input is None:
#             cms_extraction = next(
#                 (o for o in obs_list if (o.name or "").strip().lower() == "cms form extraction"),
#                 None
#             )
#             if cms_extraction:
#                 prompt_input = cms_extraction.input

#         if prompt_input is None:
#             logging.warning(
#                 f"[f2 Superbill FineTuning] No prompt observation found in trace "
#                 f"'{matched_trace.id}' for file '{file_name}'."
#             )

#         return {
#             "trace_id": matched_trace.id,
#             "trace_name": matched_trace.name,
#             "file_name": file_name,
#             "prompt": prompt_input,
#         }
#     except Exception as e:
#         logging.warning(f"[f2 Superbill FineTuning] Failed to fetch Langfuse trace for '{file_name}': {e}")
#         return None


# def process_finetuning_healthcare_superbill(payload: dict):
#     logging.info("[f2 Healthcare Superbill FineTuning] Processing payload...")
#     environment = payload.get("environment", "exp")
#     records = payload.get("records", [])
#     folder_name = str(payload.get("folder_name") or "").strip()

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

#     # Load Azure configurations for Upload Destination
#     container_name = get_azure_setting("FINAL_DATA_SUPERBILL_CONTAINER", "superbill-dataset")
#     dest_connection_string = get_azure_setting("AZURE_STORAGE_CONNECTION_STRING") 

#     # Load Azure configurations for Fallback Source Download
#     source_container_name = get_azure_setting("SOURCE_DATA_SUPERBILL_CONTAINER")
#     source_connection_string = get_azure_setting("HEALTHCARE_AI_STORAGE_CONNECTION_STRING")

#     # Initialize Destination Client (For Uploading)
#     blob_service_client = None
#     if dest_connection_string:
#         try:
#             blob_service_client = BlobServiceClient.from_connection_string(dest_connection_string)
#         except Exception as e:
#             logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Destination BlobServiceClient: {e}")
#     else:
#         logging.warning("[f2 Healthcare Superbill FineTuning] Destination connection string missing. Azure Blob uploads will be skipped.")

#     # Initialize Source Client (For Fallback Downloading)
#     source_blob_service_client = None
#     if source_connection_string:
#         try:
#             source_blob_service_client = BlobServiceClient.from_connection_string(source_connection_string)
#         except Exception as e:
#             logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Source BlobServiceClient: {e}")
#     else:
#         logging.warning("[f2 Healthcare Superbill FineTuning] Source connection string missing. Fallback downloads from blob will fail.")

#     # Initialize Langfuse Client (For fetching pipeline traces/prompts)
#     langfuse_client = None
#     if Langfuse is not None:
#         lf_public_key = get_azure_setting("HEALTHCARE_LANGFUSE_PUBLIC_KEY")
#         lf_secret_key = get_azure_setting("HEALTHCARE_LANGFUSE_SECRET_KEY")
#         lf_host = get_azure_setting("LANGFUSE_HOST", "https://cloud.langfuse.com")
#         if lf_public_key and lf_secret_key:
#             try:
#                 langfuse_client = Langfuse(
#                     public_key=lf_public_key,
#                     secret_key=lf_secret_key,
#                     host=lf_host
#                 )
#                 logging.info("[f2 Healthcare Superbill FineTuning] Langfuse client initialized.")
#             except Exception as lf_err:
#                 logging.warning(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Langfuse client: {lf_err}")
#         else:
#             logging.warning("[f2 Healthcare Superbill FineTuning] Langfuse keys missing — prompt fetch will be skipped.")
#     else:
#         logging.warning("[f2 Healthcare Superbill FineTuning] langfuse package not installed — prompt fetch will be skipped.")


#     for rec in records:
#         file_name = rec.get("file_name") or "unknown_file.pdf"
#         alloc_id = str(rec.get("allocation_id"))
#         ground_truth = rec.get("ground_truth") or {}

#         if not ground_truth:
#             logging.warning(f"[f2 Healthcare Superbill FineTuning] Missing ground truth for allocation {alloc_id}.")
#             continue

#         # Isolate base filename to use as the folder name
#         base_name, _ = os.path.splitext(file_name)
        
#         # =================================================================
#         # Folder Structure: {folder_name}/{base_name}/{base_name}.pdf|json
#         # folder_name comes from the payload; omitted if not set
#         # =================================================================
#         _prefix = f"{folder_name}/" if folder_name else ""
#         pdf_blob_path = f"{_prefix}{base_name}/{base_name}.pdf"
#         json_blob_path = f"{_prefix}{base_name}/{base_name}.json"

#         # Read SAS URL directly from ground_truth allocation
#         file_url = ground_truth.get("Allocation", {}).get("File_url")
        
#         pdf_bytes = None  # Variable to hold the file data in memory

#         if blob_service_client and container_name:
            
#             # =================================================================
#             # STEP 1: Attempt to download via SAS URL
#             # =================================================================
#             if file_url:
#                 try:
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Attempting to download PDF from SAS URL...")
#                     res = requests.get(file_url, timeout=30)
#                     if res.status_code == 200:
#                         pdf_bytes = res.content
#                         logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully downloaded PDF via URL for allocation {alloc_id}.")
#                     else:
#                         logging.warning(f"[f2 Healthcare Superbill FineTuning] URL download failed with Status {res.status_code}. Falling back to Blob Storage.")
#                 except Exception as dl_err:
#                     logging.warning(f"[f2 Healthcare Superbill FineTuning] Error downloading PDF from URL: {dl_err}. Falling back to Blob Storage.")
#             else:
#                 logging.warning(f"[f2 Healthcare Superbill FineTuning] Allocation {alloc_id} lacks 'File_url'. Falling back to Blob Storage.")

#             # =================================================================
#             # STEP 2: FALLBACK - Download from source Azure Container if URL failed
#             # =================================================================
#             if not pdf_bytes and source_container_name and source_blob_service_client:
#                 try:
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Fetching '{file_name}' from source container '{source_container_name}'...")
#                     source_blob_client = source_blob_service_client.get_blob_client(container=source_container_name, blob=file_name)
                    
#                     # Download blob into memory
#                     pdf_bytes = source_blob_client.download_blob().readall()
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully fetched PDF from source container.")
#                 except Exception as src_err:
#                     logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to fetch PDF '{file_name}' from source container: {src_err}")

#             # =================================================================
#             # STEP 3: Upload the PDF to destination container
#             # =================================================================
#             if pdf_bytes:
#                 try:
#                     blob_client = blob_service_client.get_blob_client(container=container_name, blob=pdf_blob_path)
#                     pdf_content_settings = ContentSettings(content_type="application/pdf")
#                     blob_client.upload_blob(
#                         pdf_bytes, 
#                         overwrite=True, 
#                         content_settings=pdf_content_settings
#                     )
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded PDF for allocation {alloc_id}.")
#                 except Exception as up_err:
#                     logging.error(f"[f2 Healthcare Superbill FineTuning] Error uploading PDF for allocation {alloc_id}: {up_err}")

#                 # =============================================================
#                 # STEP 4: Upload the JSON Ground Truth
#                 # =============================================================
#                 try:
#                     gt_json_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)
#                     blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_blob_path)
#                     json_content_settings = ContentSettings(content_type="application/json")
#                     blob_client.upload_blob(
#                         gt_json_str, 
#                         overwrite=True, 
#                         content_settings=json_content_settings
#                     )
#                     logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded JSON ground truth for allocation {alloc_id}.")
#                 except Exception as json_err:
#                     logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload JSON ground truth: {json_err}")

#                 # =============================================================
#                 # STEP 5: Fetch Langfuse prompt/trace and upload alongside PDF
#                 # Always uploads — empty dict {} if trace/prompt not found
#                 # =============================================================
#                 if langfuse_client:
#                     prompt_blob_path = f"{_prefix}{base_name}/{base_name}_prompt.json"
#                     prompt_data = _fetch_prompt_from_langfuse(file_name, langfuse_client)
#                     try:
#                         prompt_json_str = json.dumps(prompt_data or {}, indent=2, ensure_ascii=False, default=str)
#                         prompt_blob_client = blob_service_client.get_blob_client(
#                             container=container_name, blob=prompt_blob_path
#                         )
#                         prompt_blob_client.upload_blob(
#                             prompt_json_str,
#                             overwrite=True,
#                             content_settings=ContentSettings(content_type="application/json")
#                         )
#                         logging.info(f"[f2 Healthcare Superbill FineTuning] Uploaded prompt JSON for '{file_name}' at '{prompt_blob_path}'.")
#                     except Exception as prompt_up_err:
#                         logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload prompt JSON for allocation {alloc_id}: {prompt_up_err}")
#             else:
#                 logging.error(f"[f2 Healthcare Superbill FineTuning] Could not obtain PDF data for allocation {alloc_id}. Skipping.")

#     logging.info(f"[f2 Healthcare Superbill FineTuning] Superbill fine-tuning batch processing completed.")


import os
import json
import logging
import requests

# Import Azure Blob Storage libraries
from azure.storage.blob import BlobServiceClient, ContentSettings

# Langfuse SDK for fetching pipeline traces/prompts
try:
    from langfuse import Langfuse
    import httpx as _httpx
except ImportError:
    Langfuse = None
    _httpx = None

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


def _fetch_prompt_from_langfuse(file_name: str, langfuse_client) -> dict | None:
    """
    Searches Medical-Billing-Pipeline traces in Langfuse, matches by
    metadata["file_name"], then navigates:
      - Gemini Extraction (GENERATION) → GenerateContent (GENERATION) → input
      - OR CMS Form Extraction (GENERATION) → input directly
    Searches only the last 90 days, capped at 5 pages (250 traces).
    Returns a dict with trace_id, trace_name, file_name, prompt input, and 
    a boolean 'has_gemini' flag, or None if the trace is not found.
    """
    import datetime
    try:
        from_ts = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)

        matched_trace = None
        MAX_PAGES = 5
        for page in range(1, MAX_PAGES + 1):
            result = langfuse_client.fetch_traces(
                name="Medical-Billing-Pipeline",
                from_timestamp=from_ts,
                limit=50,
                page=page
            )
            traces = result.data if result and result.data else []
            if not traces:
                break
            for t in traces:
                meta = t.metadata or {}
                if meta.get("file_name") == file_name:
                    matched_trace = t
                    break
            if matched_trace:
                break
            if not result.meta or page >= result.meta.total_pages:
                break

        if not matched_trace:
            logging.warning(f"[f2 Superbill FineTuning] No Langfuse trace found for file: {file_name}")
            return None

        # Fetch all observations for the matched trace
        observations = langfuse_client.fetch_observations(trace_id=matched_trace.id)
        obs_list = observations.data if observations and observations.data else []

        # Build parent → [children] map
        children_of: dict = {}
        for o in obs_list:
            parent = getattr(o, "parent_observation_id", None)
            children_of.setdefault(parent, []).append(o)

        prompt_input = None
        has_gemini = False

        # Path 1: Gemini Extraction → GenerateContent → input
        gemini_extraction = next(
            (o for o in obs_list if (o.name or "").strip().lower() == "gemini extraction"),
            None
        )
        if gemini_extraction:
            has_gemini = True
            generate_content = next(
                (o for o in children_of.get(gemini_extraction.id, [])
                 if (o.name or "").strip().lower() == "generatecontent"),
                None
            )
            if generate_content:
                prompt_input = generate_content.input
            else:
                logging.warning(
                    f"[f2 Superbill FineTuning] 'GenerateContent' not found inside "
                    f"'Gemini Extraction' for trace '{matched_trace.id}'."
                )

        # Path 2: CMS Form Extraction (GENERATION) → input directly
        if prompt_input is None:
            cms_extraction = next(
                (o for o in obs_list if (o.name or "").strip().lower() == "cms form extraction"),
                None
            )
            if cms_extraction:
                prompt_input = cms_extraction.input

        if prompt_input is None:
            logging.warning(
                f"[f2 Superbill FineTuning] No prompt observation found in trace "
                f"'{matched_trace.id}' for file '{file_name}'."
            )

        return {
            "trace_id": matched_trace.id,
            "trace_name": matched_trace.name,
            "file_name": file_name,
            "prompt": prompt_input,
            "has_gemini": has_gemini  # Added indicator to track Gemini observation presence
        }
    except Exception as e:
        logging.warning(f"[f2 Superbill FineTuning] Failed to fetch Langfuse trace for '{file_name}': {e}")
        return None


def process_finetuning_healthcare_superbill(payload: dict):
    logging.info("[f2 Healthcare Superbill FineTuning] Processing payload...")
    environment = payload.get("environment", "exp")
    records = payload.get("records", [])
    folder_name = str(payload.get("folder_name") or "").strip()

    # Extract the bifurcate toggle from payload (handle potential string representations)
    bifurcate_toggle = payload.get("bifurcate_llm_non_llm",True )
    if isinstance(bifurcate_toggle, str):
        bifurcate_llm_non_llm = bifurcate_toggle.strip().lower() == "true"
    else:
        bifurcate_llm_non_llm = bool(bifurcate_toggle)

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

    # Initialize Langfuse Client (For fetching pipeline traces/prompts)
    langfuse_client = None
    if Langfuse is not None:
        lf_public_key = get_azure_setting("HEALTHCARE_LANGFUSE_PUBLIC_KEY")
        lf_secret_key = get_azure_setting("HEALTHCARE_LANGFUSE_SECRET_KEY")
        lf_host = get_azure_setting("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if lf_public_key and lf_secret_key:
            try:
                _timeout = (
                    _httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
                    if _httpx else None
                )
                _http_client = _httpx.Client(timeout=_timeout) if (_httpx and _timeout) else None
                lf_kwargs = dict(
                    public_key=lf_public_key,
                    secret_key=lf_secret_key,
                    host=lf_host
                )
                if _http_client:
                    lf_kwargs["httpx_client"] = _http_client
                langfuse_client = Langfuse(**lf_kwargs)
                logging.info("[f2 Healthcare Superbill FineTuning] Langfuse client initialized (read timeout=60s).")
            except Exception as lf_err:
                logging.warning(f"[f2 Healthcare Superbill FineTuning] Failed to initialize Langfuse client: {lf_err}")
        else:
            logging.warning("[f2 Healthcare Superbill FineTuning] Langfuse keys missing — prompt fetch will be skipped.")
    else:
        logging.warning("[f2 Healthcare Superbill FineTuning] langfuse package not installed — prompt fetch will be skipped.")


    for rec in records:
        file_name = rec.get("file_name") or "unknown_file.pdf"
        alloc_id = str(rec.get("allocation_id"))
        ground_truth = rec.get("ground_truth") or {}

        if not ground_truth:
            logging.warning(f"[f2 Healthcare Superbill FineTuning] Missing ground truth for allocation {alloc_id}.")
            continue

        base_name, _ = os.path.splitext(file_name)

        # =================================================================
        # STEP 1: Determine LLM vs. Non-LLM status early via Langfuse
        # =================================================================
        prompt_data = None
        has_gemini = False

        if langfuse_client:
            prompt_data = _fetch_prompt_from_langfuse(file_name, langfuse_client)
            if prompt_data and prompt_data.get("has_gemini"):
                has_gemini = True
        elif bifurcate_llm_non_llm:
            logging.warning(
                f"[f2 Healthcare Superbill FineTuning] 'bifurcate_llm_non_llm' is active, "
                f"but Langfuse client is uninitialized. File routing will default to 'Non_LLM'."
            )

        # =================================================================
        # STEP 2: Configure Folder Path Structure with optional bifurcation
        # =================================================================
        _prefix = f"{folder_name}/" if folder_name else ""
        
        # Inject "LLM/" or "Non_LLM/" path segment if bifurcate toggle is true
        bifurcated_folder = ""
        if bifurcate_llm_non_llm:
            bifurcated_folder = "LLM/" if has_gemini else "Non_LLM/"

        pdf_blob_path = f"{_prefix}{bifurcated_folder}{base_name}/{base_name}.pdf"
        json_blob_path = f"{_prefix}{bifurcated_folder}{base_name}/{base_name}.json"
        prompt_blob_path = f"{_prefix}{bifurcated_folder}{base_name}/{base_name}_prompt.json"

        # Read SAS URL directly from ground_truth allocation
        file_url = ground_truth.get("Allocation", {}).get("File_url")
        
        pdf_bytes = None  # Variable to hold the file data in memory

        if blob_service_client and container_name:
            
            # Download via SAS URL
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

            # Fallback - Download from source Azure Container if URL failed
            if not pdf_bytes and source_container_name and source_blob_service_client:
                try:
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Fetching '{file_name}' from source container '{source_container_name}'...")
                    source_blob_client = source_blob_service_client.get_blob_client(container=source_container_name, blob=file_name)
                    pdf_bytes = source_blob_client.download_blob().readall()
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully fetched PDF from source container.")
                except Exception as src_err:
                    logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to fetch PDF '{file_name}' from source container: {src_err}")

            # Upload files if PDF data was successfully resolved
            if pdf_bytes:
                # Upload the PDF to destination container
                try:
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=pdf_blob_path)
                    pdf_content_settings = ContentSettings(content_type="application/pdf")
                    blob_client.upload_blob(
                        pdf_bytes, 
                        overwrite=True, 
                        content_settings=pdf_content_settings
                    )
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded PDF to: {pdf_blob_path}")
                except Exception as up_err:
                    logging.error(f"[f2 Healthcare Superbill FineTuning] Error uploading PDF for allocation {alloc_id}: {up_err}")

                # Upload the JSON Ground Truth
                try:
                    gt_json_str = json.dumps(ground_truth, indent=2, ensure_ascii=False)
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_blob_path)
                    json_content_settings = ContentSettings(content_type="application/json")
                    blob_client.upload_blob(
                        gt_json_str, 
                        overwrite=True, 
                        content_settings=json_content_settings
                    )
                    logging.info(f"[f2 Healthcare Superbill FineTuning] Successfully uploaded JSON ground truth to: {json_blob_path}")
                except Exception as json_err:
                    logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload JSON ground truth: {json_err}")

                # Upload the pre-fetched Langfuse prompt trace metadata
                if langfuse_client:
                    try:
                        # Create a copy and clean up internal flag before saving
                        saved_prompt_data = dict(prompt_data) if prompt_data else {}
                        saved_prompt_data.pop("has_gemini", None)

                        prompt_json_str = json.dumps(saved_prompt_data, indent=2, ensure_ascii=False, default=str)
                        prompt_blob_client = blob_service_client.get_blob_client(
                            container=container_name, blob=prompt_blob_path
                        )
                        prompt_blob_client.upload_blob(
                            prompt_json_str,
                            overwrite=True,
                            content_settings=ContentSettings(content_type="application/json")
                        )
                        logging.info(f"[f2 Healthcare Superbill FineTuning] Uploaded prompt JSON to: {prompt_blob_path}")
                    except Exception as prompt_up_err:
                        logging.error(f"[f2 Healthcare Superbill FineTuning] Failed to upload prompt JSON for allocation {alloc_id}: {prompt_up_err}")
            else:
                logging.error(f"[f2 Healthcare Superbill FineTuning] Could not obtain PDF data for allocation {alloc_id}. Skipping upload.")

    logging.info(f"[f2 Healthcare Superbill FineTuning] Superbill fine-tuning batch processing completed.")