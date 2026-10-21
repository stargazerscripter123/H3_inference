"""Portable CPU regression tests: synthetic metadata only, no upstream tree/Git/GPU/media.

Run from any checkout with: python -m unittest discover -s omni-deployment/tests
-p test_omni_cache_contract.py -v
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))
import omni_cache_evidence as e
import omni_control as control
import omni_source_stack as stack
import run_cache_eval as runner

def events(rank,enabled=True,diagnostic=True):
    rows=[]
    for i in range(4):
        value={'schema_version':1,'rank':rank,'ranks':2,'request_ordinal':i+1,'enabled':enabled,'diagnostic':diagnostic,
               'decision':'pending' if enabled else 'off','hits':None,'placement_calls':dict.fromkeys(e.PLACEMENTS,0),
               'dit_forward_count':0 if diagnostic else None,'event':'begin','completed':None}
        rows.append(value)
        end=copy.deepcopy(value);hit=enabled and i>0
        end.update(event='end',completed=True,decision='hit' if hit else 'miss' if enabled else 'off',
                   hits=2 if hit else 0 if enabled else None,placement_calls=dict.fromkeys(e.PLACEMENTS,0 if hit else 1),
                   dit_forward_count=8 if diagnostic else None)
        rows.append(end)
    return rows


def diagnostic_value(variant, protocol_sha, plan_sha, decoder_sha, stack):
    return {'collector':'collect_cache_eval','run_status':'completed','protocol_validated':True,
        'protocol_sha256':protocol_sha,
        'optimization':{'variant':variant,'source_stack_contract_sha256':stack['stack_contract_sha256'],
            'decoder_sha256':decoder_sha,'selected_cases':['case-01'],'sp_profile':True,'formal_timing_eligible':False},
        'jobs':[{'case_id':'case-01','policy_id':'cap480','status':'passed','plan_sha256':plan_sha,
            'service_final_state':'STOPPED','cleanup_error_count':0,
            'validation_reasons':['diagnostic_run_excluded_from_formal_timing'],
            'runtime_recipe':{'validated':True,'source_stack':stack,
                'actual':{'decoder_sha256':decoder_sha,'sp_profile':True,'stage_logging':True,'nvfp4_mm_backend':'auto',
                    'nvfp4_mm_row_split':'off','vae_offload':True,'expandable_segments':True,'resize_policy':'cap',
                    'max_short_edge':480,'conditioning_cache_fastpath':variant=='cache-on'},
                'cache_evidence':{'validated':True,'instrumented':True,
                    'rank_rows':{str(rank):events(rank,variant=='cache-on',True)[1::2] for rank in (0,1)}}},
            'requests':[{'seed':seed,'phase':'warmup' if i==0 else 'measured','outcome':'succeeded',
                         'media_validated_by_client':True,'included':False,
                         'exclusion_reasons':['diagnostic_run_excluded_from_formal_timing']} for i,seed in enumerate((9000,1234,5678,9012))]}]}

class EventTests(unittest.TestCase):
    def test_both_arms_and_modes_have_four_exact_rank_bound_requests(self):
        for enabled in (False,True):
            for diagnostic in (False,True):
                rows={rank:[e.parse_event(e.MARKER+json.dumps(v)) for v in events(rank,enabled,diagnostic)] for rank in (0,1)}
                self.assertEqual(len(e.validate_events(rows,enabled=enabled,diagnostic=diagnostic)[0]),4)

    def test_missing_duplicate_mixed_hit_wrong_forward_and_bool_count_rejected(self):
        original={rank:events(rank) for rank in (0,1)}
        cases=[]
        cases.append({0:original[0]})
        cases.append({0:original[0]*2,1:original[1]})
        for key,value in (('hits',1),('dit_forward_count',7),('rank',0)):
            rows=copy.deepcopy(original);rows[1][3][key]=value;cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):e.validate_events(rows,enabled=True,diagnostic=True)
        value=events(0)[1];value['hits']=False
        with self.assertRaises(ValueError):e.parse_event(e.MARKER+json.dumps(value))

    def test_owned_pid_rank_log_binding_and_duplicate_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve(); token='a'*32
            logs=root/f'h3-omni-1000-{token[:8]}/session_001/logs';logs.mkdir(parents=True)
            paths=[]
            for rank,pid in enumerate((101,102)):
                path=logs/f'worker-abc-def-{pid}.err';paths.append(path)
                path.write_text('PRIVATE_UNKNOWN_LOG\n'+'\n'.join(e.MARKER+json.dumps(v) for v in events(rank)))
            service={'owner_token':token,'uid':1000};log=root/'service.log'
            reader=SimpleNamespace(worker_pids={log:{101,102}})
            def collect():return e.collect_cache_events(service,log,reader,enabled=True,diagnostic=True,identity_verified=True,tmp_root=root)
            result=collect();self.assertTrue(result['validated']);self.assertNotIn('PRIVATE',json.dumps(result));self.assertNotIn(token,json.dumps(result))
            (logs/'worker-abc-def-101.out').write_text(paths[0].read_text())
            self.assertFalse(collect()['validated'])

class ControllerRunnerTests(unittest.TestCase):
    def base(self):
        return {'H3_OMNI_ROOT':'/home/isaac/workdir/h3','OMNI_MODEL_DIR':'/model',
                'OMNI_ADALN_TABLE':'/adaln','OMNI_DECODER':'/decoder','OMNI_FASTVIDEO_SRC':'/synthetic/source'}

    def test_default_native_reference_cache_off_and_shell_cannot_enable(self):
        env=self.base()|{'FASTVIDEO_H3_CONDITIONING_CACHE_FASTPATH':'1'}
        with patch.dict(os.environ,env,clear=True):
            config,explicit,_=control.settings('sp2')
            child=control.child_environment(explicit)
        self.assertEqual(explicit['OMNI_SOURCE_PATCH_STACK'],'reference')
        self.assertEqual(child['FASTVIDEO_H3_CONDITIONING_CACHE_FASTPATH'],'0')
        self.assertEqual(explicit['FASTVIDEO_H3_SP_PROFILE'],'0')
        self.assertEqual(explicit['FASTVIDEO_NVFP4_MM_BACKEND'],'auto')
        self.assertNotIn('h3_reference_image_resize_policy',config['generator']['pipeline']['experimental'])

    def test_explicit_on_requires_stack_cap480_sp2_and_strict_boolean(self):
        env=self.base()|{'OMNI_SOURCE_PATCH_STACK':'reference-cache','OMNI_CONDITIONING_CACHE_FASTPATH':'1',
            'OMNI_REFERENCE_IMAGE_RESIZE_POLICY':'cap','OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE':'480'}
        with patch.dict(os.environ,env,clear=True):
            _,explicit,_=control.settings('sp2')
            self.assertEqual(explicit['FASTVIDEO_H3_CONDITIONING_CACHE_FASTPATH'],'1')
            with self.assertRaises(ValueError):control.settings('sp1')
        for key,value in [('OMNI_SOURCE_PATCH_STACK','reference'),('OMNI_CONDITIONING_CACHE_FASTPATH','true'),
                ('OMNI_REFERENCE_IMAGE_MAX_SHORT_EDGE','720'),('OMNI_MEMO_ENTRIES','0')]:
            with self.subTest(key=key),patch.dict(os.environ,env|{key:value},clear=True),self.assertRaises(ValueError):
                control.settings('sp2')

    def test_arms_keep_same_recipe_and_formal_diagnostic_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            decoder=Path(directory)/'synthetic-asset.bin';decoder.write_bytes(b'fixture metadata')
            with patch.dict(os.environ,{'OMNI_DECODER':str(decoder)},clear=True):
                off,offmeta=runner.arm_settings('off','diagnostic');on,onmeta=runner.arm_settings('on','diagnostic')
                formal,meta=runner.arm_settings('on','formal')
        self.assertEqual({k for k in off if off[k]!=on[k]},{'OMNI_CONDITIONING_CACHE_FASTPATH'})
        self.assertEqual(offmeta['source_stack_contract_sha256'],onmeta['source_stack_contract_sha256'])
        self.assertFalse(onmeta['formal_timing_eligible']);self.assertTrue(onmeta['audit_instrumentation'])
        self.assertTrue(meta['formal_timing_eligible']);self.assertFalse(meta['audit_instrumentation'])
        self.assertEqual(formal['OMNI_SP_PROFILE'],'0')

    def test_formal_requires_two_complete_instrumented_same_stack_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();folder=root/'run/omni';folder.mkdir(parents=True)
            source={'stack_contract_sha256':stack.CONTRACT_SHA256}
            metadata={'source_stack':source,'decoder_sha256':'d'*64}
            kwargs=dict(root=root,protocol_sha256='a'*64,plan_sha256='b'*64,metadata=metadata)
            paths=[]
            for variant in ('cache-off','cache-on'):
                value=diagnostic_value(variant,'a'*64,'b'*64,'d'*64,source)
                path=folder/(variant+'.json');path.write_text(json.dumps(value));paths.append(path)
            self.assertEqual(len(runner.verify_diagnostic_pair(paths,**kwargs)),2)
            value['jobs'][0]['runtime_recipe']['cache_evidence']['instrumented']=False
            paths[1].write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'mismatch'):runner.verify_diagnostic_pair(paths,**kwargs)
            with self.assertRaisesRegex(ValueError,'pair'):runner.verify_diagnostic_pair(None,**kwargs)

    def test_unknown_idle_state_never_starts_service_or_sends_signals(self):
        instance=runner.CacheRunner.__new__(runner.CacheRunner)
        instance.root=PACKAGE;instance.environment=lambda job:{};instance.save=lambda:None
        job={'status':'not_run'}
        with patch.object(runner.IdleProbe,'assert_idle',side_effect=ValueError('unknown')), \
                patch.object(runner.optimization.OptimizationRunner,'run_job') as generate:
            self.assertFalse(instance.run_job(job));generate.assert_not_called()
        self.assertEqual(job['failure_stage'],'preflight')

    def test_post_stop_idle_failure_aborts_remaining_cases(self):
        instance=runner.CacheRunner.__new__(runner.CacheRunner)
        instance.root=PACKAGE;instance.environment=lambda job:{};instance.save=lambda:None;instance.interrupted=False
        job={'status':'passed','service_final_state':'STOPPED','cleanup_errors':[]}
        with patch.object(runner.IdleProbe,'assert_idle'), \
                patch.object(runner.IdleProbe,'wait_idle_after',side_effect=ValueError('unknown')), \
                patch.object(runner.optimization.OptimizationRunner,'run_job',return_value=True):
            self.assertFalse(instance.run_job(job))
        self.assertEqual(job['status'],'failed');self.assertEqual(len(job['cleanup_errors']),1)


class PortableStackTests(unittest.TestCase):
    def synthetic_tree(self, root):
        files = {}
        for index in range(8):
            relative = 'synthetic/file%d.py' % index
            raw = ('# synthetic source %d\n' % index).encode()
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            files[relative] = {'upstream': '0' * 64, 'reference': '1' * 64,
                               'reference-cache': stack.digest(raw)}
        return {'upstream_commit': 'a' * 40, 'states': ['reference', 'reference-cache'],
                'files': files, 'artifacts': {}}

    def inspect(self, root, contract, **changes):
        args = dict(upstream_head=contract['upstream_commit'], changed_paths=list(contract['files']),
                    expected_stack='reference-cache')
        args.update(changes)
        with patch.object(stack, 'load_contract', return_value=contract):
            return stack.inspect_tree(root, **args)

    def test_real_packaged_contract_and_both_patch_artifacts_are_hash_bound(self):
        contract = stack.load_contract()
        self.assertEqual(len(contract['files']), 8)
        self.assertEqual(contract['upstream_commit'], 'cdda1b7535fe32699335e39db82c1d4db21ceb88')
        self.assertEqual(len(contract['artifacts']), 4)

    def test_packaged_contract_or_artifact_tamper_rejected(self):
        contract = stack.load_contract()
        for filename in ('stack-contract.json', 'conditioning-cache-placement-v2.patch'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                for name in ('stack-contract.json', *contract['artifacts']):
                    shutil.copyfile(stack.HERE / name, root / name)
                with (root / filename).open('ab') as stream:
                    stream.write(b'\n')
                with self.assertRaisesRegex(ValueError, 'sha256_mismatch'):
                    stack.load_contract(root)

    def test_synthetic_full_tree_accepts_exact_eight_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            contract = self.synthetic_tree(root)
            self.assertEqual(len(self.inspect(root, contract)['files']), 8)

    def test_wrong_head_dirty_path_partial_stack_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            contract = self.synthetic_tree(root)
            for kwargs in ({'upstream_head': 'b' * 40}, {'changed_paths': list(contract['files'])[:-1]},
                           {'changed_paths': [*contract['files'], 'unknown.py']}, {'expected_stack': 'reference'}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.inspect(root, contract, **kwargs)
            (root / next(iter(contract['files']))).write_text('# modified\n')
            with self.assertRaisesRegex(ValueError, 'source_stack_hash'):
                self.inspect(root, contract)

    def test_symlink_file_and_ancestor_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'source'
            contract = self.synthetic_tree(source)
            alias = root / 'alias'
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'nonaliased'):
                self.inspect(alias, contract)
            target = source / next(iter(contract['files']))
            target.unlink()
            target.symlink_to(source / list(contract['files'])[1])
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.inspect(source, contract)


class DiagnosticEvidenceTests(unittest.TestCase):
    def fixture(self):
        source = {'stack_contract_sha256': stack.CONTRACT_SHA256}
        args = dict(variant='cache-on', protocol_sha256='a' * 64, plan_sha256='b' * 64,
                    decoder_sha256='d' * 64, source_stack=source)
        value = diagnostic_value('cache-on', 'a' * 64, 'b' * 64, 'd' * 64, source)
        return value, args

    def test_exact_two_rank_miss_then_hits_and_eight_forwards_accepted(self):
        value, args = self.fixture()
        self.assertTrue(e.validate_diagnostic_evidence(value, **args))

    def test_protocol_plan_decoder_stack_and_missing_rank_rejected(self):
        value, args = self.fixture()
        for key in ('protocol_sha256', 'plan_sha256', 'decoder_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                e.validate_diagnostic_evidence(value, **(args | {key: 'e' * 64}))
        with self.assertRaises(ValueError):
            e.validate_diagnostic_evidence(value, **(args | {'source_stack': {'stack_contract_sha256': 'e' * 64}}))
        del value['jobs'][0]['runtime_recipe']['cache_evidence']['rank_rows']['1']
        with self.assertRaises(ValueError):
            e.validate_diagnostic_evidence(value, **args)

    def test_extra_exclusions_cannot_be_hidden_by_diagnostic_label(self):
        for where in ('job', 'request'):
            value, args = self.fixture()
            job = value['jobs'][0]
            (job['validation_reasons'] if where == 'job' else job['requests'][1]['exclusion_reasons']).append(
                'request_reference_hashes_unverified')
            with self.subTest(where=where), self.assertRaises(ValueError):
                e.validate_diagnostic_evidence(value, **args)


if __name__ == '__main__':
    unittest.main()

