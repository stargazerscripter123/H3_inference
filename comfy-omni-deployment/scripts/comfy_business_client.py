#!/usr/bin/env python3
"""Private-protocol ComfyUI comparison client; never retries a submitted graph.

Run on the GPU host beside the owned server. Metadata output excludes prompts,
raw execution messages and Comfy prompt IDs. No service lifecycle operations.

Use --protocol plus its known SHA256, --case-id, --assets package/pins.json,
--service-receipt saved controller READY JSON, --input-root/--comfy-output-root
equal to the controller directories, --models-root to ComfyUI/models and a
new --output-root under the project. Source images default to project/inputs.
Only the selected cap480 plan runs: first seed9000 then1234/5678/9012. Start a
fresh owned service for each case to compare first requests. Dependencies:
Pillow, websocket-client and ffprobe; existing omni-deployment must be beside
this package for the shared media metadata validator. No graph is printed.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

COMFY_PIN = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
SEEDS = [9000, 1234, 5678, 9012]
EXPECTED = {"width": 480, "height": 832, "num_frames": 362,
            "fps": 24, "num_inference_steps": 8, "guidance_scale": 1.0}
ASSET_FOLDERS = {"model": "diffusion_models", "clip": "text_encoders",
                 "vae": "vae", "audio_vae": "vae"}


class ContractError(ValueError):
    """Only stable, non-private error codes may be put in this exception."""


def require(condition, code):
    if not condition:
        raise ContractError(code)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def checked_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid_sha256")
    return value


def confined(value, root, *, exists=True):
    require(isinstance(value, (str, Path)) and "\x00" not in str(value), "invalid_path")
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve(strict=exists)
    require(path.is_relative_to(root) and path != root, "path_outside_allowed_root")
    return path


def relative_name(value):
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
            "invalid_relative_name")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(x not in {".", "..", ""} for x in value.split("/")),
            "unsafe_relative_name")
    return value


def read_object(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "json_object_required")
    return value


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def normalize_server(value):
    p = urllib.parse.urlsplit(value)
    require(p.scheme == "http" and p.hostname in {"127.0.0.1", "localhost", "::1"}
            and p.port is not None and not p.username and not p.password
            and not p.query and not p.fragment and p.path in {"", "/"}, "loopback_origin_required")
    return urllib.parse.urlunsplit((p.scheme, p.netloc, "", "", ""))


def load_case(protocol_path, protocol_sha256, case_id, root, input_root):
    """Read the sealed old protocol but select only its cap480 business plan."""
    require(case_id in {"case-01", "case-02", "case-03"}, "invalid_case_id")
    protocol_path = confined(protocol_path, root)
    require(sha_file(protocol_path) == checked_sha(protocol_sha256), "protocol_sha256_mismatch")
    protocol = read_object(protocol_path)
    require(protocol.get("schema_version") == 1, "unsupported_protocol_schema")
    require(all(protocol.get("expected", {}).get(k) == v for k, v in EXPECTED.items()),
            "protocol_output_contract_mismatch")
    cases = [c for c in protocol.get("cases", []) if c.get("case_id") == case_id]
    jobs = [j for j in protocol.get("jobs", []) if j.get("case_id") == case_id
            and j.get("policy_id") == "cap480"]
    require(len(cases) == len(jobs) == 1, "unique_cap480_case_required")
    case, job = cases[0], jobs[0]
    require(job.get("resize_policy") == "cap" and job.get("max_short_edge") == 480,
            "cap480_policy_required")
    plan_path = confined(job["plan"], root)
    require(sha_file(plan_path) == checked_sha(job["plan_sha256"]), "plan_sha256_mismatch")
    plan = read_object(plan_path)
    requests = plan.get("requests")
    require(isinstance(requests, list) and len(requests) == 4, "four_requests_required")
    refs = case.get("references")
    require(isinstance(refs, list) and 1 <= len(refs) <= 8, "one_to_eight_references_required")
    checked_sha(case.get("prompt_sha256"))
    prepared_refs = []
    for ref in refs:
        path = confined(ref["remote_path"], input_root)
        require(path.is_file(), "reference_not_regular_file")
        require(sha_file(path) == checked_sha(ref.get("sha256")), "reference_sha256_mismatch")
        require(all(type(ref.get(k)) is int and ref[k] >= 32 for k in ("width", "height")),
                "invalid_reference_dimensions")
        prepared_refs.append({"source": path, "name": path.relative_to(Path(input_root).resolve()).as_posix(),
                              "sha256": ref["sha256"], "width": ref["width"], "height": ref["height"]})
    for index, request in enumerate(requests):
        require(request.get("case_id") == case_id and type(request.get("seed")) is int
                and request["seed"] == SEEDS[index], "case_or_seed_mismatch")
        require(request.get("phase") == ("warmup" if index == 0 else "measured"), "phase_order_mismatch")
        prompt = request.get("prompt")
        require(isinstance(prompt, str) and prompt.strip()
                and hashlib.sha256(prompt.encode()).hexdigest() == case["prompt_sha256"], "prompt_sha256_mismatch")
        require((request.get("width"), request.get("height"), request.get("frames")) == (480, 832, 362),
                "request_geometry_mismatch")
        actual_refs = request.get("references", [])
        require(len(actual_refs) == len(prepared_refs), "reference_count_mismatch")
        for actual, prepared in zip(actual_refs, prepared_refs, strict=True):
            require(actual.get("type") == "image" and confined(actual["source"], input_root) == prepared["source"],
                    "reference_order_mismatch")
    return {"case_id": case_id, "prompt_sha256": case["prompt_sha256"], "references": prepared_refs,
            "requests": requests, "plan_sha256": job["plan_sha256"], "protocol_sha256": protocol_sha256}


def load_assets(path, models_root):
    manifest = read_object(path)
    require(manifest.get("schema_version") == 1 and manifest.get("comfyui", {}).get("commit") == COMFY_PIN,
            "asset_manifest_pin_mismatch")
    components = {"dit": "model", "text_encoder_with_vision": "clip", "video_vae": "vae", "audio_vae": "audio_vae"}
    assets = {"comfy_commit": COMFY_PIN, "manifest_sha256": sha_file(path)}
    models = manifest.get("models", [])
    require(isinstance(models, list) and len(models) == 4, "four_model_assets_required")
    for entry in models:
        kind = components.get(entry.get("component"))
        require(kind is not None and kind not in assets, "invalid_or_duplicate_asset_component")
        relative = relative_name(entry.get("comfy_models_relative_path"))
        require(relative.startswith(ASSET_FOLDERS[kind] + "/"), "incorrect_asset_folder")
        assets[kind] = {"name": relative.split("/", 1)[1], "sha256": checked_sha(entry.get("sha256")),
                        "bytes": entry.get("size_bytes")}
    for kind, folder in ASSET_FOLDERS.items():
        item = assets.get(kind, {})
        name = relative_name(item.get("name"))
        target = confined(folder + "/" + name, models_root)
        require(target.is_file() and type(item.get("bytes")) is int and target.stat().st_size == item["bytes"]
                and sha_file(target) == checked_sha(item.get("sha256")),
                "asset_sha256_mismatch_" + kind)
    return assets


def event_evidence(events, history, prompt_id, sampler_id):
    """Use only matching events; cached sampler output can never count as inference."""
    executed, cached, failed = set(), set(), False
    for event in events:
        data = event.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue
        if event.get("type") == "executing" and data.get("node") is not None:
            executed.add(str(data["node"]))
        if event.get("type") == "execution_cached":
            cached.update(str(node) for node in data.get("nodes", []))
        if event.get("type") in {"execution_error", "execution_interrupted"}:
            failed = True
    status = history.get("status", {})
    for message in status.get("messages", []):
        if isinstance(message, (list, tuple)) and message and message[0] in {"execution_error", "execution_interrupted"}:
            failed = True
    successful = status.get("completed") is True and status.get("status_str") == "success" and not failed
    return {"sampler_executing_observed": sampler_id in executed, "sampler_cached": sampler_id in cached,
            "history_success": successful, "execution_error_observed": failed,
            "executed_nodes": sorted(executed), "cached_nodes": sorted(cached),
            "sampling_execution_verified": successful and sampler_id in executed and sampler_id not in cached}


def output_file(history, save_id, output_root, prefix):
    """Accept exactly one newly named MP4 produced by our output node."""
    node = history.get("outputs", {}).get(save_id, {})
    entries = [v for key in ("images", "videos", "gifs") for v in node.get(key, [])
               if isinstance(v, dict) and isinstance(v.get("filename"), str) and v["filename"].lower().endswith(".mp4")]
    require(len(entries) == 1, "exactly_one_mp4_output_required")
    value = entries[0]
    require(value.get("type") == "output", "output_type_required")
    name = relative_name(value["filename"])
    require("/" not in name, "output_filename_must_be_basename")
    subfolder = value.get("subfolder", "")
    if subfolder:
        relative_name(subfolder)
    relative = (PurePosixPath(subfolder) / name).as_posix()
    require(relative.startswith(prefix + "_") or relative == prefix + ".mp4", "unexpected_output_prefix")
    path = confined(relative, output_root)
    require(path.is_file() and path.stat().st_size > 0, "empty_or_missing_output")
    return path


def summarize(rows):
    good = [r for r in rows if r.get("included")]
    warm = [r["timings"]["client_to_media_s"] for r in good if r["phase"] == "measured"]
    cold = [r["timings"]["client_to_media_s"] for r in good if r["phase"] == "warmup"]
    return {"cold_first_request_s": cold[0] if len(cold) == 1 else None,
            "warm": {"n": len(warm), "median_s": statistics.median(warm) if warm else None,
                     "min_s": min(warm) if warm else None, "max_s": max(warm) if warm else None},
            "valid": len(good), "failed": sum(r.get("status") == "failed" for r in rows),
            "not_run": sum(r.get("status") == "not_run" for r in rows)}


def read_json(server, path, timeout, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        require(len(raw) <= 32 * 1024 * 1024, "oversized_http_response")
        value = json.loads(raw)
    except urllib.error.HTTPError as error:
        raise ContractError("http_status_" + str(error.code)) from None
    require(isinstance(value, dict), "http_object_required")
    return value


def cap480_size(width, height):
    require(type(width) is int and type(height) is int and min(width, height) >= 32,
            "invalid_reference_dimensions")
    require(max(width, height) <= 4 * min(width, height), "reference_aspect_outside_contract")
    short = min(width, height)
    aligned = min(short, 480) // 32 * 32
    return width * aligned // (short * 32) * 32, height * aligned // (short * 32) * 32


def prepare_references(refs, input_root, directory):
    """GPU-host preprocessing only; images are never displayed or sent to an AI.

    Match FastVideo's cap patch: EXIF transpose, RGB, floor32 canvas, Pillow
    Lanczos. PNG preserves the resulting RGB bytes for Comfy LoadImage.
    """
    from PIL import Image, ImageOps
    directory = confined(directory, input_root, exists=False)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = []
    for index, ref in enumerate(refs):
        require(sha_file(ref["source"]) == ref["sha256"], "reference_changed_before_preparation")
        with Image.open(ref["source"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            require(image.size == (ref["width"], ref["height"]), "oriented_reference_dimensions_mismatch")
            width, height = cap480_size(*image.size)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            target = directory / f"ref-{index:02d}.png"
            # New file handle prevents overwrite even if directory contents change.
            with target.open("xb") as stream:
                image.save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
        output.append({"index": index, "name": target.relative_to(Path(input_root).resolve()).as_posix(),
                       "source_sha256": ref["sha256"], "derived_sha256": sha_file(target),
                       "source_width": ref["width"], "source_height": ref["height"],
                       "width": width, "height": height})
    return output


SAMPLER_ID, SAVE_ID = "10", "14"


def build_graph(prompt, references, seed, assets, prefix):
    """Pure fixed-recipe API graph for ComfyUI@b1a247c; no filesystem access."""
    require(isinstance(prompt, str) and prompt.strip(), "nonempty_prompt_required")
    require(type(seed) is int and 0 <= seed < 2**64, "invalid_seed")
    require(isinstance(references, list) and 1 <= len(references) <= 8, "invalid_reference_count")
    relative_name(prefix)
    require("%" not in prefix, "dynamic_output_prefix_forbidden")
    names = {kind: relative_name(assets[kind]["name"]) for kind in ASSET_FOLDERS}
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "1": node("UNETLoader", unet_name=names["model"], weight_dtype="default"),
        "2": node("MiniMaxH3SigmaShift", model=["1", 0], shift_video=12.0, shift_audio=3.0),
        "3": node("CLIPLoader", clip_name=names["clip"], type="minimax", device="default"),
        "4": node("VAELoader", vae_name=names["vae"]),
        "5": node("VAELoader", vae_name=names["audio_vae"]),
        "6": node("MiniMaxH3ReferenceToVideo", clip=["3", 0], vae=["4", 0], audio_vae=["5", 0],
                  prompt=prompt, width=480, height=832, length=362, ref_image_size="max"),
        "7": node("BasicGuider", model=["2", 0], conditioning=["6", 0]),
        "8": node("KSamplerSelect", sampler_name="euler"),
        "9": node("BasicScheduler", model=["2", 0], scheduler="simple", steps=8, denoise=1.0),
        SAMPLER_ID: node("SamplerCustomAdvanced", noise=["15", 0], guider=["7", 0], sampler=["8", 0],
                         sigmas=["9", 0], latent_image=["6", 1]),
        "11": node("VAEDecode", samples=[SAMPLER_ID, 0], vae=["4", 0]),
        "12": node("VAEDecodeAudio", samples=[SAMPLER_ID, 0], vae=["5", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=24.0,
                   bit_depth=8, color_space="sRGB", codec="none"),
        SAVE_ID: node("SaveVideo", video=["13", 0], filename_prefix=prefix,
                      format="mp4", **{"format.codec": "h264", "format.codec.encoding": "auto"}),
        "15": node("RandomNoise", noise_seed=seed),
    }
    for index, ref in enumerate(references):
        require(type(ref.get("width")) is int and type(ref.get("height")) is int
                and ref["width"] % 32 == ref["height"] % 32 == 0
                and 32 <= min(ref["width"], ref["height"]) <= 480, "prepared_reference_not_cap480")
        key = str(100 + index)
        graph[key] = node("LoadImage", image=relative_name(ref["name"]))
        graph["6"]["inputs"][f"ref_images.ref_image_{index}"] = [key, 0]
    return graph


def verify_node_inventory(graph, inventory):
    missing = sorted({node["class_type"] for node in graph.values()} - set(inventory))
    require(not missing, "required_comfy_nodes_missing")
    return {"classes": sorted({node["class_type"] for node in graph.values()}),
            "object_info_sha256": sha_value(inventory)}


def verify_service_receipt(receipt, server, input_root, output_root):
    """Require an owned READY snapshot; root still owns the live lifecycle."""
    state = receipt.get("state", receipt)
    require(isinstance(state, dict) and receipt.get("status", state.get("status")) == "READY",
            "ready_service_receipt_required")
    require(state.get("comfy_commit") == COMFY_PIN and state.get("disable_metadata") is True,
            "service_pin_or_private_metadata_mismatch")
    require(str(state.get("gpu")) in {"0", "1"}, "single_gpu_baseline_required")
    require(state.get("port") == urllib.parse.urlsplit(server).port, "service_port_mismatch")
    require(Path(state.get("input_dir", "")).resolve() == Path(input_root).resolve()
            and Path(state.get("output_dir", "")).resolve() == Path(output_root).resolve(), "service_directory_mismatch")
    require(type(state.get("started_at")) in (int, float) and math.isfinite(state["started_at"])
            and state["started_at"] <= time.time(), "service_start_time_invalid")
    require(type(state.get("pid")) is int and state["pid"] > 0 and state.get("owner_token"),
            "owned_service_identity_required")
    return {"receipt_sha256": sha_value(receipt), "comfy_commit": COMFY_PIN,
            "disable_metadata": True, "gpu": int(state["gpu"]), "port": state["port"],
            "started_at": state["started_at"], "pid": state["pid"],
            "scope": "owned READY snapshot; root lifecycle evidence must establish a fresh service per case"}


def validate_media(path, ffprobe):
    # Reuse the existing strict metadata contract, without its private request
    # manifest or raw ffprobe payload in this client's output.
    module_dir = Path(__file__).resolve().parents[2] / "omni-deployment/scripts"
    sys.path.insert(0, str(module_dir))
    from run_omni import validate_media as validate
    media = validate(path, EXPECTED, ffprobe=ffprobe)
    probe = media.pop("ffprobe")
    tags = [probe.get("format", {}).get("tags", {})]
    tags.extend(stream.get("tags", {}) for stream in probe.get("streams", []))
    for mapping in tags:
        for key, value in mapping.items():
            require(not any(word in key.lower() for word in ("prompt", "workflow"))
                    and not re.search(r'"(?:prompt|workflow)"\s*:', str(value), re.IGNORECASE),
                    "private_graph_embedded_in_media_disable_metadata_required")
    return media


def run_request(server, graph, directory, comfy_output_root, prefix, *, timeout=1800,
                ffprobe="ffprobe", transport=read_json, websocket_factory=None,
                media_validator=validate_media):
    """Socket subscribes before POST; elapsed ends after a complete local copy."""
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "invalid_timeout")
    server = normalize_server(server)
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {"status": "submitting", "included": False, "graph_sha256": sha_value(graph),
              "timings": {}, "execution_evidence": None, "server_may_still_be_running": False}
    client_id, prompt_id, socket = uuid.uuid4().hex, None, None
    events = []
    started = None
    try:
        if websocket_factory is None:
            import websocket
            websocket_factory = websocket.create_connection
        socket = websocket_factory(server.replace("http://", "ws://", 1) + "/ws?clientId=" + client_id,
                                   timeout=min(timeout, 30), enable_multithread=True)
        started = time.perf_counter()
        deadline = started + timeout
        def budget():
            remaining = deadline - time.perf_counter()
            require(remaining > 0, "request_deadline_exceeded")
            return min(remaining, 30.0)
        submitted = transport(server, "/prompt", budget(), {"prompt": graph, "client_id": client_id})
        prompt_id = submitted.get("prompt_id")
        require(isinstance(prompt_id, str) and prompt_id and not submitted.get("node_errors"), "submission_not_accepted")
        record["prompt_id_sha256"] = hashlib.sha256(prompt_id.encode()).hexdigest()
        record["timings"]["client_submit_s"] = time.perf_counter() - started
        record["status"] = "executing"
        record["server_may_still_be_running"] = True
        save_json(directory / "manifest.json", record)
        finished = False
        while not finished:
            socket.settimeout(budget())
            try:
                raw = socket.recv()
            except TimeoutError:
                continue
            except Exception as error:
                # websocket-client uses its own timeout exception class.
                if type(error).__name__ == "WebSocketTimeoutException":
                    continue
                raise
            if isinstance(raw, bytes):
                continue  # Binary preview pixels are discarded, never stored/decoded.
            require(isinstance(raw, str) and len(raw) <= 16 * 1024 * 1024, "invalid_websocket_message")
            event = json.loads(raw)
            if not isinstance(event, dict) or not isinstance(event.get("data"), dict):
                continue
            data, kind = event["data"], event.get("type")
            if data.get("prompt_id") != prompt_id:
                continue
            # Keep only node IDs and event kinds; errors may contain the full prompt.
            events.append({"type": kind, "data": {key: data[key] for key in ("prompt_id", "node", "nodes") if key in data}})
            if kind in {"execution_error", "execution_interrupted"}:
                record["server_may_still_be_running"] = False
                raise ContractError("comfy_execution_failed")
            finished = kind == "execution_success" or (kind == "executing" and data.get("node") is None)
        record["server_may_still_be_running"] = False
        # A completion socket event can precede history insertion; bounded poll.
        history = None
        while history is None:
            histories = transport(server, "/history/" + urllib.parse.quote(prompt_id, safe=""), budget())
            candidate = histories.get(prompt_id)
            if isinstance(candidate, dict):
                history = candidate
                break
            time.sleep(min(0.05, budget()))
        recorded = history.get("prompt", [])
        require(isinstance(recorded, (list, tuple)) and len(recorded) >= 3
                and recorded[1] == prompt_id and recorded[2] == graph, "history_graph_identity_mismatch")
        evidence = event_evidence(events, history, prompt_id, SAMPLER_ID)
        record["execution_evidence"] = evidence
        require(evidence["sampling_execution_verified"], "sampling_execution_unverified_or_cached")
        source = output_file(history, SAVE_ID, comfy_output_root, prefix)
        target = directory / "output.mp4"
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        record["timings"]["client_to_media_s"] = time.perf_counter() - started
        record["status"] = "validating"
        audit_start = time.perf_counter()
        media = media_validator(target, ffprobe)
        require(media.get("valid") is True and checked_sha(media.get("sha256")) == sha_file(source), "delivered_media_sha_mismatch")
        record["timings"]["local_validation_s"] = time.perf_counter() - audit_start
        record.update(status="completed", included=True, output={"path": str(target), **media})
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "failed"
        record["failure_code"] = str(error) if isinstance(error, ContractError) else type(error).__name__
        if started is not None:
            record["timings"]["client_elapsed_s"] = time.perf_counter() - started
        if prompt_id is None and started is not None:
            record["server_may_still_be_running"] = True
    finally:
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        if record["execution_evidence"] is None:
            record["execution_evidence"] = event_evidence(events, {}, prompt_id, SAMPLER_ID)
        save_json(directory / "manifest.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--case-id", choices=["case-01", "case-02", "case-03"], required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/isaac/workdir/h3"))
    parser.add_argument("--input-root", type=Path, required=True, help="must equal ComfyUI --input-directory")
    parser.add_argument("--reference-root", type=Path, help="source references; defaults to project inputs")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--service-receipt", type=Path, required=True, help="controller status JSON, READY")
    parser.add_argument("--output-root", type=Path, required=True, help="parent for a new client report directory")
    parser.add_argument("--comfy-output-root", type=Path, required=True, help="must equal ComfyUI --output-directory")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        server = normalize_server(args.server)
        require(shutil.which(args.ffprobe) is not None, "ffprobe_missing")
        require(math.isfinite(args.timeout) and args.timeout > 0, "invalid_timeout")
        service = verify_service_receipt(read_object(args.service_receipt), server, args.input_root, args.comfy_output_root)
        for path in (args.input_root, args.models_root, args.output_root, args.comfy_output_root):
            confined(path, args.root, exists=False)
        reference_root = args.reference_root or args.root / "inputs"
        confined(reference_root, args.root)
        case = load_case(args.protocol, args.protocol_sha256, args.case_id, args.root, reference_root)
        assets = load_assets(args.assets, args.models_root)
        token = uuid.uuid4().hex
        root = args.output_root.resolve() / (args.case_id + "_" + token)
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        refs = prepare_references(case["references"], args.input_root, "omni-comfy-business-" + token)
        rows = [{"phase": r["phase"], "seed": r["seed"], "status": "not_run", "included": False}
                for r in case["requests"]]
        report = {"schema_version": 1, "backend": "comfyui-native", "case_id": args.case_id,
                  "status": "running", "protocol_sha256": case["protocol_sha256"], "plan_sha256": case["plan_sha256"],
                  "prompt_sha256": case["prompt_sha256"], "comfy_commit": COMFY_PIN,
                  "assets_manifest_sha256": assets["manifest_sha256"],
                  "service": service,
                  "assets": {k: {"name": assets[k]["name"], "sha256": assets[k]["sha256"]} for k in ASSET_FOLDERS},
                  "references": refs, "requests": rows, "output_contract": EXPECTED,
                  "started_at": time.time(),
                  "timing_scope": "POST submission through successful history and complete local MP4 copy+fsync; preparation and validation excluded",
                  "cold_scope": "first request of this client case; fresh service and OS/GPU coldness require external lifecycle evidence",
                  "quality_status": "pending_human_review",
                  "comparison_limitations": ["Native Comfy video VAE FP16 differs from FastVideo NVFP4 36L decoder; whole-recipe comparison.",
                      "Matching input RGB preparation does not prove identical framework conditioning, color handling or numerical output.",
                      "Sampler executing evidence establishes node execution, not an instrumented count of eight DiT forwards."]}
        for index, request in enumerate(case["requests"]):
            prefix = "omni-business/" + token + "/" + args.case_id + "_" + str(request["seed"])
            graph = build_graph(request["prompt"], refs, request["seed"], assets, prefix)
            if index == 0:
                report["node_inventory"] = verify_node_inventory(graph, read_json(server, "/object_info", 30))
            rows[index]["status"] = "in_progress"
            report["summary"] = summarize(rows)
            save_json(root / "report.json", report)
            result = run_request(server, graph, root / f"request-{index:02d}", args.comfy_output_root,
                                 prefix, timeout=args.timeout, ffprobe=args.ffprobe)
            rows[index].update(result)
            report["summary"] = summarize(rows)
            save_json(root / "report.json", report)
            if not result["included"]:
                break
        report["status"] = "completed" if all(r["included"] for r in rows) else "failed"
        report["finished_at"] = time.time()
        report["summary"] = summarize(rows)
        save_json(root / "report.json", report)
        print(json.dumps({"report": str(root / "report.json"), "status": report["status"], "summary": report["summary"]}))
        return 0 if report["status"] == "completed" else 1
    except (Exception, KeyboardInterrupt) as error:
        code = str(error) if isinstance(error, ContractError) else type(error).__name__
        if "report" in locals():
            for row in report["requests"]:
                if row["status"] == "in_progress":
                    row.update(status="failed", included=False, failure_code=code)
            report.update(status="failed", failure_code=code, finished_at=time.time(), summary=summarize(report["requests"]))
            save_json(root / "report.json", report)
        print(json.dumps({"status": "failed", "failure_code": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
