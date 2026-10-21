#!/usr/bin/env python3
"""Run one explicit optimization variant against the sealed cap480 business plans.

The original nine-job protocol is fully validated and never rewritten. Each
selected case keeps its original plan, prompt, ordered references and four seeds.
Use a new record and output directory for every attempt, including diagnostics.
This entry point inherits the owned lifecycle, cleanup and no-retry behavior of
run_business_eval.py; it does not relax that runner's baseline contract.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

import run_business_eval as business

VARIANTS = {
    "baseline": {"backend": "auto", "vae_offload": True, "decoder": "baseline"},
    "cutlass": {"backend": "cutlass", "vae_offload": True, "decoder": "baseline"},
    "cudnn": {"backend": "cudnn", "vae_offload": True, "decoder": "baseline"},
    "vae-resident": {"backend": "auto", "vae_offload": False, "decoder": "baseline"},
    "decoder8": {"backend": "auto", "vae_offload": True, "decoder": "decoder8"},
}


def select_jobs(protocol: dict, prepared: dict, case_id: str | None) -> tuple[dict, dict]:
    """Select without mutating the fully validated source protocol or plans."""
    business.require(case_id is None or case_id in {"case-01", "case-02", "case-03"}, "invalid_case_selection")
    jobs = [dict(job) for job in protocol["jobs"]
            if job["policy_id"] == "cap480" and (case_id is None or job["case_id"] == case_id)]
    business.require(len(jobs) == (3 if case_id is None else 1), "cap480_jobs_missing")
    selected = {}
    for job in jobs:
        info = prepared[job["job_id"]]
        requests = business.read_object(info["plan"])["requests"]
        business.require([item["phase"] for item in requests] == ["warmup", "measured", "measured", "measured"],
                         "optimization_requires_one_first_and_three_warm_requests")
        business.require(len({item["seed"] for item in requests}) == 4, "optimization_requires_four_distinct_seeds")
        selected[job["job_id"]] = info
    return {**protocol, "jobs": jobs}, selected


def variant_settings(variant: str, sp_profile: int, stage_logging: int) -> tuple[dict, dict]:
    business.require(variant in VARIANTS, "invalid_optimization_variant")
    business.require(type(sp_profile) is int and sp_profile in {0, 1}, "invalid_sp_profile")
    business.require(type(stage_logging) is int and stage_logging in {0, 1}, "invalid_stage_logging")
    recipe = VARIANTS[variant]
    asset_name = "OMNI_DECODER8" if recipe["decoder"] == "decoder8" else "OMNI_DECODER"
    decoder_text = os.environ.get(asset_name, "")
    decoder = Path(decoder_text).expanduser()
    business.require(bool(decoder_text) and decoder.is_absolute() and decoder.is_file(),
                     "verified_absolute_decoder_asset_required")
    decoder = decoder.resolve()
    env = {"OMNI_NVFP4_MM_BACKEND": recipe["backend"], "OMNI_STAGE_LOGGING": str(stage_logging),
           "OMNI_VAE_OFFLOAD": "1" if recipe["vae_offload"] else "0", "OMNI_EXPANDABLE_SEGMENTS": "1",
           "OMNI_SP_PROFILE": str(sp_profile), "OMNI_DECODER": str(decoder), "OMNI_GPU_IDS": "0,1",
           "OMNI_MEMO_ENTRIES": "16"}
    metadata = {"schema_version": 1, "variant": variant, "policy_id": "cap480", "profile": "sp2",
                "nvfp4_mm_backend": recipe["backend"], "nvfp4_mm_row_split": "off",
                "vae_offload": recipe["vae_offload"], "expandable_segments": True,
                "stage_logging": bool(stage_logging), "sp_profile": bool(sp_profile),
                "formal_timing_eligible": not bool(sp_profile), "decoder_variant": recipe["decoder"],
                "decoder_path": str(decoder), "decoder_sha256": business.sha_file(decoder),
                "requests_per_case": 4, "first_requests_per_case": 1, "warm_requests_per_case": 3,
                "memo_entries": 16,
                "notes": ["Original protocol and plan SHA256 identities are retained; only cap480 jobs are selected.",
                          "First request and warm requests must be reported separately.",
                          "SP_PROFILE=1 is diagnostic and must be excluded from formal timing comparisons.",
                          "Decoder layer count and provenance require independent asset verification; the variant label alone is not proof.",
                          "Expected overrides are not proof of actual launch; verify the controller identity in every service log."]}
    return env, metadata


class OptimizationRunner(business.Runner):
    def __init__(self, *args, overrides: dict, metadata: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.overrides = dict(overrides)
        self.record["optimization"] = metadata
        self.record["optimization_runner_sha256"] = business.sha_file(Path(__file__))
        self.record["base_runner_sha256"] = business.sha_file(Path(business.__file__))
        for job in self.record["jobs"]:
            job["optimization_variant"] = metadata["variant"]

    def environment(self, job: dict) -> dict:
        env = super().environment(job)
        env.update(self.overrides)
        return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/home/isaac/workdir/h3"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--case", choices=("case-01", "case-02", "case-03"))
    parser.add_argument("--sp-profile", type=int, choices=(0, 1), default=0,
                        help="1 marks the entire attempt diagnostic, ineligible for formal timing")
    parser.add_argument("--stage-logging", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        business.require(sys.platform == "linux", "linux_required_for_owned_service_lifecycle")
        root = args.project_root.expanduser().resolve()
        protocol_path = args.protocol.expanduser().resolve()
        record_path = business.within(args.record, root / "run/omni")
        output = business.within(args.output_root, root / "outputs")
        business.require(not record_path.exists() and not output.exists(), "new_record_and_output_paths_required")
        protocol, prepared = business.validate_protocol(protocol_path, root)
        selected, prepared = select_jobs(protocol, prepared, args.case)
        overrides, metadata = variant_settings(args.variant, args.sp_profile, args.stage_logging)
        metadata["selected_cases"] = [job["case_id"] for job in selected["jobs"]]
        runner = OptimizationRunner(root, record_path, output, protocol_path, selected, prepared,
                                    overrides=overrides, metadata=metadata)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        fd = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)

        def interrupt(_signum, _frame):
            runner.interrupted = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        code = runner.run()
        print(business.json.dumps({"status": runner.record["status"], "record": str(record_path),
                                   "variant": args.variant, "formal_timing_eligible": not bool(args.sp_profile)}))
        return code
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(business.json.dumps({"status": "preflight_failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
