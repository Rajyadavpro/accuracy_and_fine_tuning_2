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
# Main processor — called by the function app router
# payload is already the unpacked body dict (router strips the outer "body" wrapper)
# ---------------------------------------------------------------------------

def process_finetuning_ccai(payload: dict):
    logging.info("[f2 CCAI FineTuning] Processing payload...")

    # 1. Validate environment variables
    ccai_conn_string = os.getenv("CCAI_STORAGE_CONNECTION_STRING")
    azure_conn_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    target_container_name = os.getenv("FINAL_DATA_CCAI_CONTAINER")

    if not all([ccai_conn_string, azure_conn_string, target_container_name]):
        logging.error("[f2 CCAI FineTuning] Missing one or more required environment variables: "
                      "CCAI_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONNECTION_STRING, FINAL_DATA_CCAI_CONTAINER")
        return

    # 2. Parse payload — payload is the inner body dict
    source_container_name = payload.get("container")
    blob_names = payload.get("blob_names") or []

    if not source_container_name or not blob_names:
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
            target_container_name=target_container_name
        )

    logging.info(f"[f2 CCAI FineTuning] All {len(blob_names)} items processed.")


def _process_single_blob(
    source_container_name: str,
    folder_prefix: str,
    ccai_conn_string: str,
    azure_conn_string: str,
    target_container_name: str
) -> None:
    """Process a single blob folder (folder_prefix)."""
    logging.info(f"[f2 CCAI FineTuning] Processing folder: {source_container_name}/{folder_prefix}")

    # Set up blob clients
    source_container_client = BlobServiceClient.from_connection_string(ccai_conn_string) \
        .get_container_client(source_container_name)
    target_container_client = BlobServiceClient.from_connection_string(azure_conn_string) \
        .get_container_client(target_container_name)

    if not target_container_client.exists():
        target_container_client.create_container()
        logging.info(f"[f2 CCAI FineTuning] Created target container: {target_container_name}")

    # Use /tmp for ephemeral local storage (safe in Azure Functions)
    tmp_dir = os.getenv("TEMP", "/tmp")
    original_transcript_path = os.path.join(tmp_dir, f"{folder_prefix}_transcript.json")
    original_audio_path = os.path.join(tmp_dir, f"{folder_prefix}_recording.ogg")
    modified_json_path = os.path.join(tmp_dir, f"{folder_prefix}_modified.json")
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
            text = current_turn["text"]
            start_time_str = current_turn["timestamp"]
            start_ms = _time_to_ms(start_time_str)

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
                "clip_name": clip_name
            })

            logging.info(f"[f2 CCAI FineTuning] Clipping {clip_name} ({start_time_str} → {end_time_str})")
            audio[start_ms:end_ms].export(clip_filename, format="ogg")
            clipped_files.append(clip_filename)

        # Save modified transcript JSON
        with open(modified_json_path, "w", encoding="utf-8") as f:
            json.dump({"turns": modified_turns}, f, indent=4)

        # Upload all files to target container
        files_to_upload = [
            (original_transcript_path, "call_transcript.json"),
            (original_audio_path, "call_recording.ogg"),
            (modified_json_path, "modified_json.json"),
        ] + [(p, os.path.basename(p)) for p in clipped_files]

        logging.info(f"[f2 CCAI FineTuning] Uploading {len(files_to_upload)} files to {target_container_name}/{folder_prefix}/")
        for local_path, blob_name in files_to_upload:
            blob_client = target_container_client.get_blob_client(f"{folder_prefix}/{blob_name}")
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logging.info(f"[f2 CCAI FineTuning] Uploaded: {folder_prefix}/{blob_name}")

        logging.info(f"[f2 CCAI FineTuning] Completed for {folder_prefix}.")

    except Exception as e:
        logging.error(f"[f2 CCAI FineTuning] Processing failed for {folder_prefix}: {e}", exc_info=True)

    finally:
        # Clean up all temp files for this folder
        for fpath in [original_transcript_path, original_audio_path, modified_json_path] + clipped_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as cleanup_err:
                    logging.warning(f"[f2 CCAI FineTuning] Failed to cleanup {fpath}: {cleanup_err}")