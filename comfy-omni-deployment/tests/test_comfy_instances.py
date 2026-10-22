"""Two-slot CPU tests: no network, GPU, real subprocesses, signals or media IO."""
from pathlib import Path
from contextlib import contextmanager
import io
import json
import os
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import comfy_control as c
import comfy_service as worker
import comfy_business_client as client


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.configs={slot:c.configuration(self.root,instance=slot) for slot in ('default','gpu0','gpu1')}
        for cfg in self.configs.values():Path(cfg['run_dir']).mkdir(parents=True,exist_ok=True)

    def state(self,slot,pid=1234,token='a'*32):
        cfg=self.configs[slot]
        return {'pid':pid,'pgid':pid,'uid':os.getuid(),'start_ticks':str(pid+7),'owner_token':token,
                'service_script':str(SCRIPTS/'comfy_service.py'),'config':str(Path(cfg['run_dir'])/'launch.json'),
                'gpu':cfg['gpu'],'port':cfg['port'],'instance':slot,'identity_established':True,
                'fingerprint':c.fingerprint(cfg),'comfy_commit':c.PIN,'input_dir':cfg['input_dir'],
                'output_dir':cfg['output_dir'],'disable_metadata':True,'started_at':1}

    def save(self,state,slot=None):
        run=Path(self.configs[slot or state['instance']]['run_dir'])
        c.atomic_json(run/'service.json',state)
        return run

    def test_default_compatible_and_fixed_slot_paths_ports_gpus(self):
        default=self.configs['default']
        self.assertNotIn('instance',default)
        self.assertEqual((default['gpu'],default['port']),(1,8188))
        self.assertEqual(default['run_dir'],str(self.root/'run/comfy-omni'))
        self.assertEqual(c.configuration(self.root,gpu=0)['port'],8188)
        for slot,gpu,port in (('gpu0',0,8189),('gpu1',1,8190)):
            cfg=self.configs[slot]
            self.assertEqual((cfg['gpu'],cfg['port']),(gpu,port))
            for key in ('run_dir','input_dir','user_dir','temp_base'):
                self.assertIn('/instances/'+slot,cfg[key])
            self.assertEqual(cfg['output_dir'],str(self.root/'outputs/comfy-omni/instances'/slot))
            self.assertEqual(c.child_environment(cfg)['CUDA_VISIBLE_DEVICES'],str(gpu))
            launch={**cfg,'owner_token':'a'*32,'temp_dir':str(Path(cfg['temp_base'])/('a'*32))}
            c.validate_config(launch)
            argv=c.command_arguments(launch)
            self.assertEqual(argv[argv.index('--port')+1],str(port))
        self.assertEqual(len({cfg['user_dir'] for cfg in self.configs.values()}),3)

    def test_cross_slot_gpu_io_alias_and_port_tampering_are_rejected(self):
        for kwargs in ({'gpu':1,'instance':'gpu0'}, {'instance':'unregistered'},
                       {'instance':'gpu0','input_dir':self.configs['gpu1']['input_dir']},
                       {'instance':'gpu0','input_dir':str(self.root/'inputs')},
                       {'instance':'gpu0','output_dir':self.configs['gpu1']['output_dir']}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):c.configuration(self.root,**kwargs)
        cfg={**self.configs['gpu0'],'port':8190}
        with self.assertRaisesRegex(ValueError,'launch_configuration_mismatch'):c.validate_config(cfg)
        link=self.root/'run/comfy-omni/instances/gpu0/input'
        link.symlink_to(Path(self.configs['gpu1']['run_dir']),target_is_directory=True)
        with self.assertRaises(ValueError):c.configuration(self.root,instance='gpu0')

    def test_two_owned_gpu_slots_launch_without_overwriting_each_other(self):
        identities={}; launches=[]
        def popen(argv,**kwargs):
            pid=1234+len(launches);launches.append((argv,kwargs))
            identities[pid]={'uid':os.getuid(),'pgid':pid,'start_ticks':str(pid+7),
                              'cmdline':[arg.encode() for arg in argv]}
            return SimpleNamespace(pid=pid,poll=lambda:None)
        with patch.object(c,'check_port') as ports,patch.object(c,'verify_installation',return_value={}), \
             patch.object(c.subprocess,'Popen',side_effect=popen),patch.object(c,'proc_identity',side_effect=lambda pid:identities.get(pid)):
            first=c.start(self.configs['gpu0'],Path(self.configs['gpu0']['run_dir']))
            before=(Path(self.configs['gpu0']['run_dir'])/'service.json').read_bytes()
            second=c.start(self.configs['gpu1'],Path(self.configs['gpu1']['run_dir']))
        self.assertEqual([call.args[0] for call in ports.call_args_list],[8189,8190])
        self.assertEqual(first['instance'],'gpu0');self.assertEqual(second['instance'],'gpu1')
        self.assertNotEqual(first['owner_token'],second['owner_token'])
        self.assertNotEqual(first['pid'],second['pid'])
        self.assertEqual((Path(self.configs['gpu0']['run_dir'])/'service.json').read_bytes(),before)
        self.assertEqual([entry[1]['env']['CUDA_VISIBLE_DEVICES'] for entry in launches],['0','1'])
        for state in (first,second):self.assertIn('/instances/'+state['instance']+'/',state['log'])

    def test_live_default_on_same_gpu_and_unknown_other_slot_block_launch_no_signals(self):
        default=self.state('default');self.save(default)
        for mode in ('same_gpu','unverified','orphaned'):
            default.update(identity_established=mode!='unverified');self.save(default)
            with self.subTest(mode=mode),patch.object(c,'owned',return_value=mode!='orphaned'), \
                 patch.object(c,'unverified_alive',return_value=True),patch.object(c,'group_members',return_value=[{'pid':9}]), \
                 patch.object(c.subprocess,'Popen') as popen,patch.object(c.os,'kill') as kill,patch.object(c.os,'killpg') as killpg:
                with self.assertRaises(RuntimeError):c.start(self.configs['gpu1'],Path(self.configs['gpu1']['run_dir']))
                popen.assert_not_called();kill.assert_not_called();killpg.assert_not_called()

    def test_receipt_from_other_slot_cannot_authorize_stop_or_status(self):
        wrong=self.state('gpu0');self.save(wrong,slot='gpu1')
        run=Path(self.configs['gpu1']['run_dir'])
        with patch.object(c,'owned') as owned,patch.object(c.os,'kill') as kill,patch.object(c.os,'killpg') as killpg:
            for action in (lambda:c.status(run),lambda:c.stop(run,wrong['owner_token'],force=True)):
                with self.assertRaisesRegex(RuntimeError,'saved_instance_scope_mismatch'):action()
            owned.assert_not_called();kill.assert_not_called();killpg.assert_not_called()

    def test_stopping_gpu0_leaves_gpu1_receipt_unchanged(self):
        first=self.state('gpu0');second=self.state('gpu1',pid=5678,token='b'*32)
        run0=self.save(first);run1=self.save(second);before=(run1/'service.json').read_bytes()
        with patch.object(c,'owned',side_effect=[True,True,False,False]),patch.object(c,'group_members',return_value=[]), \
             patch.object(c.os,'kill') as kill,patch.object(c.os,'killpg') as killpg:
            result=c.stop(run0,first['owner_token'],timeout=0)
        self.assertEqual(result['status'],'STOPPED');kill.assert_called_once_with(1234,signal.SIGTERM)
        killpg.assert_not_called();self.assertEqual((run1/'service.json').read_bytes(),before)

    def test_readiness_probes_selected_port_and_requires_single_visible_gpu(self):
        state=self.state('gpu1');urls=[]
        def opening(url,timeout):
            urls.append(url)
            body={'system':{'comfyui_version':c.VERSION},'devices':[{'type':'cuda','index':0}]} if url.endswith('system_stats') else {'MiniMaxH3ReferenceToVideo':{}}
            return io.StringIO(json.dumps(body))
        with patch.object(c,'listener_owned',return_value=True),patch.object(c.urllib.request,'build_opener',return_value=SimpleNamespace(open=opening)):
            self.assertEqual(c.probe(state)['status'],'READY')
        self.assertEqual(urls,['http://127.0.0.1:8190/system_stats','http://127.0.0.1:8190/object_info/MiniMaxH3ReferenceToVideo'])
        state['port']=8189
        with self.assertRaisesRegex(RuntimeError,'saved_instance_port_mismatch'):c.state_port(state)

    def test_client_receipt_accepts_isolated_port_but_rejects_cross_slot_endpoint(self):
        for slot in ('gpu0','gpu1'):
            state={**self.state(slot),'status':'READY'}
            receipt=client.verify_service_receipt(state,f'http://127.0.0.1:{state["port"]}',state['input_dir'],state['output_dir'])
            self.assertEqual(receipt['gpu'],state['gpu']);self.assertEqual(receipt['port'],state['port'])
            with self.assertRaisesRegex(ValueError,'service_port_mismatch'):
                client.verify_service_receipt(state,'http://127.0.0.1:8188',state['input_dir'],state['output_dir'])

    def test_cli_uses_global_start_lock_then_slot_lock(self):
        events=[]
        @contextmanager
        def global_lock(root):events.append('global');yield
        @contextmanager
        def slot_lock(run):events.append(('slot',str(run)));yield
        with patch.object(c.sys,'platform','linux'),patch.object(c,'locked_instance_starts',side_effect=global_lock), \
             patch.object(c,'locked',side_effect=slot_lock),patch.object(c,'start',return_value={'status':'STARTING'}) as start, \
             patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(c.main(['start','--project-root',str(self.root),'--instance','gpu0']),0)
        self.assertEqual(events,['global',('slot',self.configs['gpu0']['run_dir'])])
        self.assertEqual(start.call_args.args[0],self.configs['gpu0'])

if __name__=='__main__':unittest.main()
