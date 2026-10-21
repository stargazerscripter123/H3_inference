"""Read-only idle guards copied from the reviewed formal orchestrator. No signals."""
import json
from pathlib import Path
import subprocess
import time

class Abort(ValueError): pass
class RetryableIdle(Abort): pass

def require(value,code):
    if not value: raise Abort(code)

class IdleProbe:
    def __init__(self, root, environment):
        self.root=root
        self.environment=environment
        self.python=root/'envs/omni/bin/python'
        self.interrupted=False

    def run_json(self, command, *, timeout=30):
        result = subprocess.run(command, cwd=self.root, env=self.environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout, check=False)
        require(result.returncode == 0, 'status_command_failed')
        try: value = json.loads(result.stdout)
        except ValueError: raise Abort('status_response_unreadable') from None
        require(isinstance(value,dict), 'status_object_required')
        return value

    @staticmethod
    def probe_timeout(deadline, cap):
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'idle_probe_deadline_exhausted')
        return min(cap, remaining)

    def assert_idle(self, *, deadline=None):
        # Initial preflight remains one strict observation. Post-variant callers
        # share one deadline across ALL commands, not 30+30+20 seconds per retry.
        deadline = time.monotonic() + 60 if deadline is None else deadline
        comfy = self.run_json([str(self.root/'envs/comfy-omni/bin/python'),
                               str(self.root/'comfy-omni-deployment/scripts/comfy_control.py'),
                               'status','--project-root',str(self.root)],
                              timeout=self.probe_timeout(deadline, 30))
        require(comfy.get('status') == 'STOPPED', 'comfy_not_stopped_no_signals_sent')
        omni = self.run_json([str(self.python),str(self.root/'omni-deployment/scripts/omni_control.py'),'status','sp2'],
                             timeout=self.probe_timeout(deadline, 30))
        require(omni.get('status') == 'STOPPED', 'fastvideo_not_stopped_no_signals_sent')
        self.probe_timeout(deadline, 1)
        for table in ('/proc/net/tcp','/proc/net/tcp6'):
            for line in Path(table).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16) in (8108,8188):
                    raise RetryableIdle('service_listener_present_no_signals_sent')
        gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,stdin=subprocess.DEVNULL,
                             timeout=self.probe_timeout(deadline, 20),check=False)
        require(gpu.returncode == 0, 'gpu_probe_failed_no_signals_sent')
        lines = gpu.stdout.splitlines()
        require(all(line.strip().isdigit() for line in lines if line.strip()),
                'gpu_probe_response_unknown_no_signals_sent')
        self.probe_timeout(deadline, 1)
        if gpu.stdout.strip():
            raise RetryableIdle('gpu_compute_present_no_signals_sent')

    def wait_idle_after(self, row, *, timeout=60):
        require(0 < timeout <= 60, 'post_variant_idle_timeout_out_of_bounds')
        started = time.monotonic()
        deadline = started + timeout
        receipt = {'status':'checking', 'attempts':0, 'timeout_s':timeout,
                   'first_reason':None, 'last_reason':None, 'final_observation':None,
                   'required_consecutive_idle':2, 'consecutive_idle':0,
                   'poll_interval_s':2, 'scope':'read_only_no_service_signals'}
        row['post_variant_idle_check'] = receipt
        def reason(code):
            if receipt['first_reason'] is None: receipt['first_reason'] = code
            receipt['last_reason'] = code
            receipt['final_observation'] = code
        try:
            while time.monotonic() < deadline:
                require(not self.interrupted, 'orchestration_interrupted')
                receipt['attempts'] += 1
                try:
                    self.assert_idle(deadline=deadline)
                except RetryableIdle as error:
                    receipt['consecutive_idle'] = 0
                    reason(str(error))
                except (Abort,OSError,ValueError,IndexError,subprocess.SubprocessError) as error:
                    # Unknown controller/identity/JSON/probe status is never a
                    # drain condition. There is no service signal or recovery.
                    code = str(error) if isinstance(error,Abort) else 'idle_probe_' + type(error).__name__
                    reason(code)
                    receipt['status'] = 'rejected'
                    raise
                else:
                    receipt['consecutive_idle'] += 1
                    receipt['final_observation'] = 'idle'
                    if receipt['consecutive_idle'] == 2:
                        receipt['status'] = 'confirmed'
                        return
                # A second idle observation must be at least two seconds later;
                # insufficient time cannot be converted into a shorter interval.
                remaining = deadline - time.monotonic()
                if remaining < 2: break
                time.sleep(2)
            receipt['status'] = 'timed_out'
            receipt['final_observation'] = 'deadline_before_two_consecutive_idle'
            raise Abort('post_variant_idle_deadline_exhausted_no_signals_sent')
        finally:
            receipt['elapsed_s'] = round(time.monotonic() - started, 6)
