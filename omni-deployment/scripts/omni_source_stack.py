"""Exact source-stack identity and explicit stopped-service patch transitions.

The controller uses read-only identity checks under its existing lock. The CLI
may invoke Git check/apply/revert only under that lock with no service receipt.
This module does not import FastVideo, read media, or start/stop services.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "patches"
CONTRACT_SHA256 = "2aef55488c8f2f46f5993fa926c6d69e781f53cb4b33d7bd66c87f320d299c94"
STACKS = {"reference", "reference-cache"}
FLAG = "FASTVIDEO_H3_CONDITIONING_CACHE_FASTPATH"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_file(root, relative):
    root = Path(root)
    require(root.is_absolute() and root == root.resolve() and not root.is_symlink(), "explicit_nonaliased_root_required")
    path = Path(relative)
    require(not path.is_absolute() and path.parts and all(part not in {".", ".."} for part in path.parts),
            "invalid_contract_relative_path")
    target = root
    for part in path.parts:
        target /= part
        require(not target.is_symlink(), "contract_symlink_rejected")
    require(target.resolve().is_relative_to(root.resolve()) and target.is_file(), "contract_file_missing_or_outside")
    return target


def load_contract(directory=HERE):
    directory = Path(directory)
    require(directory.is_absolute() and directory == directory.resolve(), "contract_directory_alias_rejected")
    raw = safe_file(directory, "stack-contract.json").read_bytes()
    require(digest(raw) == CONTRACT_SHA256, "stack_contract_sha256_mismatch")
    contract = json.loads(raw)
    require(contract["schema_version"] == 1 and len(contract["files"]) == 8, "stack_contract_schema_invalid")
    for relative, expected in contract["artifacts"].items():
        require(digest(safe_file(directory, relative).read_bytes()) == expected, "stack_artifact_sha256_mismatch")
    reference = json.loads(safe_file(directory, "reference-image-cap-v1.json").read_text())
    cache = json.loads(safe_file(directory, "conditioning-cache-placement-v2.json").read_text())
    require(reference["upstream_commit"] == cache["upstream_commit"] == contract["upstream_commit"], "stack_upstream_mismatch")
    require(cache["requires_reference_patch_sha256"] == reference["patch_sha256"]
            == contract["artifacts"]["reference-image-cap-v1.patch"], "stack_parent_patch_mismatch")
    require(cache["patch_sha256"] == contract["artifacts"]["conditioning-cache-placement-v2.patch"], "cache_patch_mismatch")
    for relative, hashes in contract["files"].items():
        ref, extra = reference["files"].get(relative), cache["files"].get(relative)
        if ref:
            require((hashes["upstream"], hashes["reference"]) == (ref["before_sha256"], ref["after_sha256"]),
                    "reference_lineage_mismatch")
        if extra:
            require((hashes["reference"], hashes["reference-cache"]) == (extra["before_sha256"], extra["after_sha256"]),
                    "cache_lineage_mismatch")
    return contract


def inspect_tree(source, *, upstream_head, changed_paths, expected_stack, directory=HERE):
    """Verify all eight files and reject all unexpected tracked modifications.

    upstream_head and changed_paths MUST come from the explicit checkout, not
    user-provided receipt JSON. The controller and explicit CLI acquire
    control.lock; the CLI also requires a stopped service before mutation.
    """
    contract = load_contract(directory)
    require(expected_stack in contract["states"], "unknown_stack")
    require(upstream_head == contract["upstream_commit"], "upstream_commit_mismatch")
    expected_changes = {name for name, values in contract["files"].items()
                        if values[expected_stack] != values["upstream"]}
    require(set(changed_paths) == expected_changes and len(changed_paths) == len(expected_changes),
            "unexpected_or_missing_tracked_modifications")
    actual = {}
    for relative, expected in contract["files"].items():
        actual[relative] = digest(safe_file(source, relative).read_bytes())
        require(actual[relative] == expected[expected_stack], "source_stack_hash_mismatch")
    return {"schema_version":1, "stack_id":expected_stack, "upstream_commit":upstream_head,
            "stack_contract_sha256":CONTRACT_SHA256, "artifact_sha256":contract["artifacts"], "files":actual}


def deployment_knobs(values):
    """Explicit OMNI knobs; inherited FASTVIDEO values never select a variant."""
    stack = values.get("OMNI_SOURCE_PATCH_STACK", "reference")
    flag = values.get("OMNI_CONDITIONING_CACHE_FASTPATH", "0")
    require(stack in STACKS, "invalid_omni_source_patch_stack")
    require(type(flag) is str and flag in {"0", "1"}, "cache_fastpath_requires_0_or_1")
    require(flag == "0" or stack == "reference-cache", "enabled_cache_requires_exact_cache_stack")
    return {"OMNI_SOURCE_PATCH_STACK":stack, "OMNI_CONDITIONING_CACHE_FASTPATH":flag, FLAG:flag}


def validate_trial_scope(config, env):
    """Additional gates for BOTH same-stack A/B arms; existing gates still apply."""
    engine = config["generator"]["engine"]
    pipeline = config["generator"]["pipeline"]
    experimental, offload = pipeline["experimental"], engine["offload"]
    require(env.get("OMNI_SOURCE_PATCH_STACK") == "reference-cache" and env.get(FLAG) in {"0", "1"}, "trial_stack_or_flag_unverified")
    require(all(type(value) is int for value in
                (engine["num_gpus"], engine["parallelism"]["tp_size"], engine["parallelism"]["sp_size"]))
            and engine["num_gpus"] == 2 and engine["parallelism"]["tp_size"] == 1
            and engine["parallelism"]["sp_size"] == 2 and engine["execution_backend"] == "ray"
            and engine["use_fsdp_inference"] is False, "trial_sp2_tp1_ray_required")
    require(experimental.get("h3_sequential_load") is True
            and (experimental.get("h3_encoder_split") is None or experimental.get("h3_encoder_split") is False)
            and env.get("FASTVIDEO_H3_ENCODER_SPLIT", "0") == "0"
            and offload["lazy_module_load"] is False and offload["dit"] is False
            and offload["dit_layerwise"] is False and offload["text_encoder"] is True
            and engine["compile"]["enabled"] is False, "trial_sequential_scope_mismatch")
    require(pipeline["components"].get("override_pipeline_cls_name") == "MiniMaxH3Ref2VAModularPipeline"
            and not pipeline["components"].get("lora_path"), "trial_ref2va_without_lora_required")
    require(experimental.get("h3_reference_image_resize_policy") == "cap"
            and type(experimental.get("h3_reference_image_max_short_edge")) is int
            and experimental["h3_reference_image_max_short_edge"] == 480
            and env.get("FASTVIDEO_H3_REF2VA_MEMO_ENTRIES") == "16"
            and env.get("FASTVIDEO_LOG_ALL_PROCESSES") == "1", "trial_cap480_memo16_allrank_required")
    return True


def validate_worker_decisions(rank_rows):
    """Strict numeric collector target: no inferred rank or missing-event fill.

    The companion omni_cache_evidence collector establishes unique owned Ray
    worker/rank bindings and successful request boundaries. This narrower
    helper only validates the numeric decision sequence supplied by callers.
    """
    require(type(rank_rows) is dict and all(type(rank) is int for rank in rank_rows)
            and set(rank_rows) == {0, 1}, "both_sp_ranks_required")
    expected = [("miss", 0, 2), ("hit", 2, 2), ("hit", 2, 2), ("hit", 2, 2)]
    for rows in rank_rows.values():
        require(type(rows) is list and len(rows) == 4
                and all(type(row) is tuple and len(row) == 3 and type(row[0]) is str
                        and type(row[1]) is int and type(row[2]) is int for row in rows)
                and rows == expected, "first_miss_then_three_allrank_hits_required")
    return True


def inspect_source(source, expected_stack="reference-cache"):
    import subprocess
    source = Path(source)
    require(source.is_absolute() and source == source.resolve(), "source_alias_rejected")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--name-only", "-z"])
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"])
    require(not any(Path(p.decode()).suffix in {".py", ".so", ".json"} for p in untracked.split(b"\0") if p),
            "untracked_runtime_source_rejected")
    return inspect_tree(source, upstream_head=head, changed_paths=[p.decode() for p in changes.split(b"\0") if p],
                        expected_stack=expected_stack)


def source_identity(source, config, env):
    knobs = deployment_knobs(env)
    if knobs["OMNI_SOURCE_PATCH_STACK"] == "reference":
        from omni_reference_patch import inspect_patch
        policy = config["generator"]["pipeline"]["experimental"].get("h3_reference_image_resize_policy", "native")
        return {"reference_patch": inspect_patch(source, required=policy != "native")}
    validate_trial_scope(config, env)
    return {"source_stack": inspect_source(source)}


def main():
    import argparse
    import fcntl
    import subprocess
    parser = argparse.ArgumentParser(description="Explicit exact-stack check/apply/revert; never edits a running service")
    parser.add_argument("action", choices=("check", "apply", "revert"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-stack", choices=("reference", "reference-cache"), required=True)
    args = parser.parse_args()
    root, source = args.project_root, args.source
    require(root.is_absolute() and root == root.resolve() and source.is_absolute()
            and source == source.resolve() and source.is_relative_to(root / "src"), "explicit_owned_source_required")
    run = root / "run/omni"
    run.mkdir(parents=True, exist_ok=True)
    with (run / "control.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = inspect_source(source, args.expected_stack)
        if args.action != "check":
            require(not (run / "service.json").exists(), "clean_service_stop_required_before_source_mutation")
            before, after = (("reference", "reference-cache") if args.action == "apply" else ("reference-cache", "reference"))
            require(args.expected_stack == before, "unexpected_patch_transition")
            command = ["git", "-C", str(source), "apply"]
            if args.action == "revert": command.append("--reverse")
            patch = HERE / "conditioning-cache-placement-v2.patch"
            subprocess.run(command + ["--check", str(patch)], check=True)
            subprocess.run(command + [str(patch)], check=True)
            identity = inspect_source(source, after)
        print(json.dumps(identity, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
