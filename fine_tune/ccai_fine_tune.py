import json
import logging
import os

# Add ffmpeg directory to PATH BEFORE importing pydub
# This ensures pydub can find ffmpeg and ffprobe automatically
_ffmpeg_bin_paths = [
    r"C:\Users\raj.kumaryadav\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
]

for bin_path in _ffmpeg_bin_paths:
    if os.path.isdir(bin_path):
        os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
        break

from pydub import AudioSegment
from azure.storage.blob import BlobServiceClient
from langfuse import Langfuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_to_ms(time_str: str) -> int:
    h, m, s = time_str.split(':')
    sec, ms = s.split('.')
    return (int(h) * 3600000) + (int(m) * 60000) + (int(sec) * 1000) + int(ms)


def _ms_to_time(ms: int) -> str:
    h = int(ms // 3600000)
    m = int((ms % 3600000) // 60000)
    s = int((ms % 60000) // 1000)
    milli = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


# ---------------------------------------------------------------------------
# Langfuse helper
# ---------------------------------------------------------------------------

def _fetch_llm_data_from_langfuse(folder_prefix: str, langfuse_client) -> dict:
    """
    For a given folder_prefix (session UUID), finds the agent_session trace in
    Langfuse and extracts every agent_turn observation (input + output).
    Returns a dict ready to be serialised as llm_data.json.
    """
    result = {
        "session_id": folder_prefix,
        "trace_id": None,
        "turns": []
    }

    if not langfuse_client:
        return result

    try:
        trace_id = None

        # 1. Search by session_id where trace name == "agent_session"
        search = langfuse_client.fetch_traces(
            name="agent_session",
            session_id=folder_prefix,
            limit=1
        )
        traces = search.data if search and search.data else []
        if traces:
            trace_id = traces[0].id
        else:
            # 2. Fallback: treat the folder_prefix itself as a trace_id
            try:
                t = langfuse_client.get_trace(folder_prefix)
                if t and getattr(t, "name", None) == "agent_session":
                    trace_id = t.id
            except Exception:
                pass

        if not trace_id:
            logging.warning(
                f"[f2 CCAI FineTuning] No agent_session trace found for session: {folder_prefix}"
            )
            return result

        result["trace_id"] = trace_id

        # 3. Collect ALL observations in the trace
        all_observations = []
        page = 1
        while True:
            obs_result = langfuse_client.fetch_observations(
                trace_id=trace_id,
                limit=100,
                page=page
            )
            observations = obs_result.data if obs_result and obs_result.data else []
            if not observations:
                break
            all_observations.extend(observations)
            if len(observations) < 100:
                break
            page += 1

        # Sort by start_time so turns appear in conversation order
        all_observations.sort(key=lambda o: getattr(o, "start_time", None) or "")

        # Build parent_id → [children] map so we can find the GENERATION
        # child that lives inside each agent_turn span
        children_map: dict = {}
        for obs in all_observations:
            parent_id = getattr(obs, "parent_observation_id", None)
            if parent_id:
                children_map.setdefault(parent_id, []).append(obs)

        # 4. Walk every agent_turn span; the LLM input/output is in the
        #    GENERATION child (or in the span itself if it IS a generation)
        for obs in all_observations:
            if getattr(obs, "name", None) != "agent_turn":
                continue

        def _find_descendant(parent_id: str, name: str):
            """BFS through children_map to find the first descendant with the given name."""
            queue = list(children_map.get(parent_id, []))
            while queue:
                node = queue.pop(0)
                if getattr(node, "name", None) == name:
                    return node
                queue.extend(children_map.get(node.id, []))
            return None

        # 4. Walk every agent_turn span; drill down to llm_request for the
        #    actual LLM prompt (input) and response (output)
        for obs in all_observations:
            if getattr(obs, "name", None) != "agent_turn":
                continue

            # agent_turn → llm_node → llm_request
            llm_request = _find_descendant(obs.id, "llm_request")
            source = llm_request if llm_request is not None else obs

            raw_input = source.input
            raw_output = source.output

            # --- normalise input to a list of {role, content} message dicts ---
            if isinstance(raw_input, list):
                messages = raw_input
            elif isinstance(raw_input, dict):
                messages = (
                    raw_input.get("messages")
                    or raw_input.get("input")
                    or [raw_input]
                )
            else:
                messages = []

            # --- normalise output to a plain response string ---
            if isinstance(raw_output, str):
                response = raw_output
            elif isinstance(raw_output, dict):
                response = (
                    raw_output.get("content")
                    or raw_output.get("output")
                    or raw_output.get("text")
                    or str(raw_output)
                )
            elif isinstance(raw_output, list) and raw_output:
                first = raw_output[0]
                if isinstance(first, dict):
                    response = first.get("content") or first.get("text") or str(first)
                else:
                    response = str(first)
            else:
                response = str(raw_output) if raw_output is not None else ""

            result["turns"].append({
                "observation_id": obs.id,
                "input": messages,
                "output": response,
            })

        logging.info(
            f"[f2 CCAI FineTuning] Fetched {len(result['turns'])} agent_turn(s) for {folder_prefix}"
        )

    except Exception as e:
        logging.warning(
            f"[f2 CCAI FineTuning] Failed to fetch Langfuse LLM data for {folder_prefix}: {e}"
        )

    return result


# ---------------------------------------------------------------------------
# Main processor — called by the function app router
# payload is already the unpacked body dict (router strips the outer "body" wrapper)
# ---------------------------------------------------------------------------

def process_finetuning_ccai(payload: dict):
    logging.info("[f2 CCAI FineTuning] Processing payload...")

    # 1. Validate environment variables
    ccai_conn_string = os.getenv("CCAI_STORAGE_CONNECTION_STRING")
    azure_conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    source_container_name = os.getenv("SOURCE_DATA_CCAI_CONTAINER")

    if not all([ccai_conn_string, azure_conn_string, source_container_name]):
        logging.error("[f2 CCAI FineTuning] Missing one or more required environment variables: "
                      "CCAI_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONNECTION_STRING, SOURCE_DATA_CCAI_CONTAINER")
        return

    # 2a. Initialise Langfuse client for agent_session / agent_turn lookup
    langfuse_client = None
    ccai_lf_pub = os.getenv("CCAI_LANGFUSE_PUBLIC_KEY")
    ccai_lf_sec = os.getenv("CCAI_LANGFUSE_SECRET_KEY")
    lf_host = os.getenv("LANGFUSE_HOST")
    if ccai_lf_pub and ccai_lf_sec:
        try:
            langfuse_client = Langfuse(
                public_key=ccai_lf_pub,
                secret_key=ccai_lf_sec,
                host=lf_host
            )
        except Exception as lf_err:
            logging.warning(f"[f2 CCAI FineTuning] Langfuse init failed: {lf_err}")

    # 2. Parse payload — payload is the inner body dict
    # 'container' in the payload is the final/target container where processed data is saved
    target_container_name = payload.get("container")
    blob_names = payload.get("blob_names") or []
    folder_name = str(payload.get("folder_name") or "").strip()

    if not target_container_name or not blob_names:
        logging.error("[f2 CCAI FineTuning] Payload missing required fields: 'container' and/or 'blob_names'.")
        return

    # 3. Process each blob_name (folder_prefix) in the array
    for blob_name in blob_names:
        folder_prefix = blob_name.split("/")[0] if "/" in blob_name else blob_name
        logging.info(f"[f2 CCAI FineTuning] Processing blob_name: {blob_name} (folder: {folder_prefix})")
        _process_single_blob(
            source_container_name=source_container_name,
            folder_prefix=folder_prefix,
            ccai_conn_string=ccai_conn_string,
            azure_conn_string=azure_conn_string,
            target_container_name=target_container_name,
            folder_name=folder_name,
            langfuse_client=langfuse_client
        )

    logging.info(f"[f2 CCAI FineTuning] All {len(blob_names)} items processed.")


def _process_single_blob(
    source_container_name: str,
    folder_prefix: str,
    ccai_conn_string: str,
    azure_conn_string: str,
    target_container_name: str,
    folder_name: str = "",
    langfuse_client=None
) -> None:
    """Process a single blob folder (folder_prefix)."""
    logging.info(f"[f2 CCAI FineTuning] Processing folder: {source_container_name}/{folder_prefix}")

    # Set up blob clients
    source_container_client = BlobServiceClient.from_connection_string(ccai_conn_string) \
        .get_container_client(source_container_name)
    target_container_client = BlobServiceClient.from_connection_string(azure_conn_string) \
        .get_container_client(target_container_name)

    if not source_container_client.exists():
        logging.error(
            f"[f2 CCAI FineTuning] Source container '{source_container_name}' does not exist. "
            f"Skipping folder: {folder_prefix}"
        )
        return

    if not target_container_client.exists():
        target_container_client.create_container()
        logging.info(f"[f2 CCAI FineTuning] Created target container: {target_container_name}")

    # Use /tmp for ephemeral local storage (safe in Azure Functions)
    tmp_dir = os.getenv("TEMP", "/tmp")
    original_transcript_path = os.path.join(tmp_dir, f"{folder_prefix}_transcript.json")
    original_audio_path = os.path.join(tmp_dir, f"{folder_prefix}_recording.ogg")
    modified_json_path = os.path.join(tmp_dir, f"{folder_prefix}_modified.json")
    llm_data_path = os.path.join(tmp_dir, f"{folder_prefix}_llm_data.json")
    clipped_files = []

    try:
        # Download source files
        logging.info(f"[f2 CCAI FineTuning] Downloading transcript and audio for {folder_prefix}...")
        with open(original_transcript_path, "wb") as f:
            f.write(source_container_client.download_blob(f"{folder_prefix}/call_transcript.json").readall())

        with open(original_audio_path, "wb") as f:
            f.write(source_container_client.download_blob(f"{folder_prefix}/call_recording.ogg").readall())

        # Load audio and transcript
        audio = AudioSegment.from_ogg(original_audio_path)
        total_audio_ms = len(audio)

        with open(original_transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        turns = transcript_data.get("turns", [])
        if not turns:
            logging.warning(f"[f2 CCAI FineTuning] Transcript for {folder_prefix} has no turns. Skipping.")
            return

        modified_turns = []

        # Clip audio per turn and build modified transcript
        for i, current_turn in enumerate(turns):
            role = current_turn["role"]
            role_normalized = str(role).strip().lower()
            text = current_turn["text"]
            start_time_str = current_turn["timestamp"]
            start_ms = _time_to_ms(start_time_str)

            if role_normalized == "agent":
                service_type = "stt"
            elif role_normalized == "customer":
                service_type = "tts"
            else:
                service_type = ""

            if i + 1 < len(turns):
                end_time_str = turns[i + 1]["timestamp"]
                end_ms = _time_to_ms(end_time_str)
            else:
                end_ms = total_audio_ms
                end_time_str = _ms_to_time(end_ms)

            clip_name = f"{role}_{i + 1}"
            clip_filename = os.path.join(tmp_dir, f"{folder_prefix}_{clip_name}.ogg")

            modified_turns.append({
                "role": role,
                "starting_timestamp": start_time_str,
                "end_timestamp": end_time_str,
                "text": text,
                "clip_name": clip_name,
                "service_type": service_type
            })

            logging.info(f"[f2 CCAI FineTuning] Clipping {clip_name} ({start_time_str} → {end_time_str})")
            audio[start_ms:end_ms].export(clip_filename, format="ogg")
            clipped_files.append(clip_filename)

        # Save modified transcript JSON
        with open(modified_json_path, "w", encoding="utf-8") as f:
            json.dump({"turns": modified_turns}, f, indent=4)

        # Fetch LLM fine-tuning data from Langfuse (agent_session → agent_turn)
        llm_data = _fetch_llm_data_from_langfuse(folder_prefix, langfuse_client)
        with open(llm_data_path, "w", encoding="utf-8") as f:
            json.dump(llm_data, f, indent=4)

        # Root files go directly under {folder_prefix}/
        # Clips go into {folder_prefix}/clips/ with cleaner names (UUID prefix stripped)
        _upload_prefix = f"{folder_name}/{folder_prefix}" if folder_name else folder_prefix
        root_files = [
            (original_transcript_path, "call_transcript.json"),
            (original_audio_path, "call_recording.ogg"),
            (modified_json_path, "modified_json.json"),
            (llm_data_path, "llm_data.json"),
        ]
        clip_files = [
            (p, f"clips/{os.path.basename(p).replace(f'{folder_prefix}_', '', 1)}")
            for p in clipped_files
        ]
        files_to_upload = root_files + clip_files

        logging.info(f"[f2 CCAI FineTuning] Uploading {len(files_to_upload)} files to {target_container_name}/{_upload_prefix}/")
        for local_path, blob_name in files_to_upload:
            blob_client = target_container_client.get_blob_client(f"{_upload_prefix}/{blob_name}")
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logging.info(f"[f2 CCAI FineTuning] Uploaded: {_upload_prefix}/{blob_name}")

        logging.info(f"[f2 CCAI FineTuning] Completed for {folder_prefix}.")

    except Exception as e:
        logging.error(f"[f2 CCAI FineTuning] Processing failed for {folder_prefix}: {e}", exc_info=True)

    finally:
        # Clean up all temp files for this folder
        for fpath in [original_transcript_path, original_audio_path, modified_json_path, llm_data_path] + clipped_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as cleanup_err:
                    logging.warning(f"[f2 CCAI FineTuning] Failed to cleanup {fpath}: {cleanup_err}")