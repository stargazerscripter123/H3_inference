"""CPU regression for the pre-ray.init address handoff; external imports mocked.

No Ray runtime, FastVideo import, subprocess, GPU, network, or media is used.
The simulated worker snapshots its inherited address at ray.init, so a later
driver-only fix cannot make this test pass.
"""
from pathlib import Path
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import omni_service


class RayStartupAddressTest(unittest.TestCase):
    chosen_address = "192.0.2.10"

    def fixture(self, source: Path, actual_node_address: str):
        names = (
            "fastvideo", "fastvideo.entrypoints", "fastvideo.entrypoints.cli",
            "fastvideo.entrypoints.cli.inference_config", "fastvideo.api",
            "fastvideo.api.compat", "fastvideo.utils", "fastvideo.entrypoints.cli.serve",
            "ray", "omni_ray_health",
        )
        modules = {name: ModuleType(name) for name in names}
        for module in modules.values():
            module.__path__ = []
        modules["fastvideo"].__file__ = str(source / "fastvideo/__init__.py")
        generator = NS(engine=NS(execution_backend="ray", num_gpus=2))
        modules["fastvideo.entrypoints.cli.inference_config"].build_serve_config = Mock(return_value=NS(generator=generator))
        modules["fastvideo.api.compat"].generator_config_to_fastvideo_args = Mock()
        events = []
        inherited_worker_environment = {}

        def choose_address():
            self.assertNotIn("FASTVIDEO_HOST_IP", os.environ)
            events.append("choose_address")
            return self.chosen_address

        def initialize(**kwargs):
            self.assertEqual(kwargs["address"], "local")
            self.assertNotIn("RAY_ADDRESS", os.environ)
            self.assertEqual(kwargs["_node_ip_address"], self.chosen_address)
            self.assertEqual(os.environ.get("FASTVIDEO_HOST_IP"), self.chosen_address)
            # Model Ray's process inheritance at initialization time, not after.
            inherited_worker_environment["FASTVIDEO_HOST_IP"] = os.environ.get("FASTVIDEO_HOST_IP")
            events.append("ray.init")

        ray = modules["ray"]
        ray.is_initialized = Mock(return_value=False)
        ray.init = Mock(side_effect=initialize)
        ray.get_runtime_context = Mock(return_value=NS(get_node_id=lambda: "owned-node"))
        ray.nodes = Mock(return_value=[{
            "Alive": True, "NodeID": "owned-node", "NodeManagerAddress": actual_node_address,
            "Resources": {f"node:{actual_node_address}": 1, "GPU": 2},
        }])
        ray.cluster_resources = Mock(return_value={"GPU": 2, "CPU": 8})
        ray.shutdown = Mock(side_effect=lambda: events.append("ray.shutdown"))
        modules["fastvideo.utils"].get_ip = Mock(side_effect=choose_address)
        health_adapter = Mock(side_effect=lambda: events.append("health_adapter"))
        modules["omni_ray_health"].install_ray_health_adapter = health_adapter
        serve = Mock(side_effect=lambda _args: events.append("serve"))
        modules["fastvideo.entrypoints.cli.serve"].ServeSubcommand = Mock(return_value=NS(cmd=serve))
        return NS(modules=modules, ray=ray, events=events, worker_env=inherited_worker_environment,
                  health_adapter=health_adapter, serve=serve)

    def invoke_main(self, source: Path, fixture):
        arguments = ["omni_service.py", "--source", str(source), "--config", "unused-config.json",
                     "--owner-token", "test-owned-runtime"]
        # Every external boundary used by main is replaced; no git command or
        # production import can escape the test. Restore sys.path after main.
        with patch.dict(sys.modules, fixture.modules), \
                patch.dict(os.environ, {"FASTVIDEO_HOST_IP": "127.0.0.1", "RAY_ADDRESS": "ray://stale.invalid:10001"}, clear=True), \
                patch.object(sys, "argv", arguments), patch.object(sys, "path", list(sys.path)), \
                patch.object(omni_service.signal, "signal"), \
                patch.object(omni_service.subprocess, "check_output", return_value=omni_service.FASTVIDEO_COMMIT + "\n"), \
                patch("builtins.print"):
            omni_service.main()

    def test_workers_inherit_chosen_address_before_ray_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            fixture = self.fixture(source, self.chosen_address)
            self.invoke_main(source, fixture)
        self.assertEqual(fixture.worker_env, {"FASTVIDEO_HOST_IP": self.chosen_address})
        self.assertEqual(fixture.events, ["choose_address", "ray.init", "health_adapter", "serve", "ray.shutdown"])
        fixture.ray.init.assert_called_once()
        fixture.ray.shutdown.assert_called_once()
        fixture.serve.assert_called_once()

    def test_normalized_address_mismatch_stops_before_serving_and_shuts_down(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            fixture = self.fixture(source, "192.0.2.99")
            with self.assertRaisesRegex(RuntimeError, "normalized the chosen node address unexpectedly"):
                self.invoke_main(source, fixture)
        self.assertEqual(fixture.worker_env, {"FASTVIDEO_HOST_IP": self.chosen_address})
        self.assertEqual(fixture.events, ["choose_address", "ray.init", "ray.shutdown"])
        fixture.health_adapter.assert_not_called()
        fixture.serve.assert_not_called()
        fixture.ray.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
