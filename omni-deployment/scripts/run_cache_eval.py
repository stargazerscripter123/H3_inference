#!/usr/bin/env python3
"""Same-stack cache placement A/B; original sealed cap480 plans remain untouched."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import run_business_eval as business
import run_optimization_eval as optimization
from omni_source_stack import CONTRACT_SHA256, inspect_source
from omni_cache_idle import IdleProbe
from omni_cache_evidence import validate_diagnostic_evidence


def arm_settings(arm, mode):
    business.require(arm in {'off', 'on'} and mode in {'diagnostic', 'formal'}, 'invalid_cache_arm_or_mode')
    env, metadata = optimization.variant_settings('baseline', int(mode == 'diagnostic'), 1)
    env.update(OMNI_SOURCE_PATCH_STACK='reference-cache', OMNI_CONDITIONING_CACHE_FASTPATH=str(int(arm == 'on')), OMNI_PORT='8108')
    metadata.update(variant='cache-' + arm, source_patch_stack='reference-cache',
                    source_stack_contract_sha256=CONTRACT_SHA256, conditioning_cache_fastpath=arm == 'on',
                    audit_instrumentation=mode == 'diagnostic')
    metadata['notes'].extend(['Both arms require identical reference+cache source bytes; only cache flag differs.',
        'Diagnostic mode counts real transformer calls by a temporary pre-hook; exclude all its latency from formal results.',
        'Formal mode emits small request/rank events but installs no transformer hook; actual NFE is unobserved there.',
        'Placement counters are method invocation counts, not measured physical PCIe transfers.'])
    return env, metadata



def verify_diagnostic_pair(paths, *, root, protocol_sha256, plan_sha256, metadata):
    business.require(isinstance(paths, list) and len(paths) == 2, 'formal_requires_same_stack_diagnostic_pair')
    variants, receipts = set(), []
    for path in paths:
        path=business.within(path, root/'run/omni')
        value=business.read_object(path)
        variant=value.get('optimization', {}).get('variant')
        business.require(variant not in variants, 'duplicate_diagnostic_arm')
        validate_diagnostic_evidence(value, variant=variant, protocol_sha256=protocol_sha256,
            plan_sha256=plan_sha256, decoder_sha256=metadata['decoder_sha256'], source_stack=metadata['source_stack'])
        variants.add(variant)
        receipts.append({'variant':variant,'evidence_path':str(path),'evidence_sha256':business.sha_file(path)})
    business.require(variants=={'cache-off','cache-on'},'both_diagnostic_arms_required')
    return receipts


class CacheRunner(optimization.OptimizationRunner):
    def run_job(self, job):
        probe=IdleProbe(self.root, self.environment(job))
        try:
            probe.assert_idle()
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
            job.update(status='failed', failure_stage='preflight', finished_at=business.now(),
                       failure={'code':'cache_initial_idle_unverified','type':type(error).__name__})
            self.save()
            return False
        job['initial_idle_check']={'status':'confirmed','checked_at':business.now(),'scope':'both_controllers_ports_gpu_read_only'}
        self.save()
        proceed=super().run_job(job)
        if job.get('service_final_state')=='STOPPED' and not job['cleanup_errors']:
            try:
                probe.interrupted=self.interrupted
                probe.wait_idle_after(job)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                job['cleanup_errors'].append('cache_post_stop_idle_unverified_no_signals')
                job['status']='failed'
                proceed=False
            self.save()
        return proceed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path('/home/isaac/workdir/h3'))
    parser.add_argument('--arm', choices=('off', 'on'), required=True)
    parser.add_argument('--mode', choices=('diagnostic', 'formal'), default='diagnostic')
    parser.add_argument('--case', choices=('case-01', 'case-02', 'case-03'), default='case-01')
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--diagnostic-evidence', type=Path, nargs=2, help='Formal mode requires collected same-stack case01 off/on diagnostics')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        business.require(sys.platform == 'linux', 'linux_required')
        root = args.project_root.resolve()
        protocol_path = args.protocol.resolve()
        record_path = business.within(args.record, root / 'run/omni')
        output = business.within(args.output_root, root / 'outputs')
        business.require(not record_path.exists() and not output.exists(), 'new_record_and_output_paths_required')
        protocol, prepared = business.validate_protocol(protocol_path, root)
        selected, prepared = optimization.select_jobs(protocol, prepared, None if args.all_cases else args.case)
        overrides, metadata = arm_settings(args.arm, args.mode)
        source = Path(os.environ.get('OMNI_FASTVIDEO_SRC', str(root / 'src/FastVideo')))
        metadata['source_stack'] = inspect_source(source)
        if args.mode == 'formal':
            metadata['diagnostic_gate'] = verify_diagnostic_pair(args.diagnostic_evidence, root=root,
                protocol_sha256=business.sha_file(protocol_path),
                plan_sha256=next(j['plan_sha256'] for j in protocol['jobs'] if j['job_id']=='case-01-cap480'), metadata=metadata)
        metadata['selected_cases'] = [job['case_id'] for job in selected['jobs']]
        runner = CacheRunner(root, record_path, output, protocol_path, selected, prepared,
                                                 overrides=overrides, metadata=metadata)
        runner.record['cache_runner_sha256'] = business.sha_file(Path(__file__))
        record_path.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        fd = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        def interrupt(_signum, _frame): runner.interrupted = True
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        code = runner.run()
        print(json.dumps({'status': runner.record['status'], 'variant': metadata['variant'],
                          'formal_timing_eligible': args.mode == 'formal', 'record': str(record_path)}))
        return code
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({'status':'preflight_failed', 'error_type':type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
