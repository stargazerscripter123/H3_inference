"""Synthetic metadata and transport tests; no image/video decoding or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("comfy_business_client", Path(__file__).resolve().parents[1] / "scripts/comfy_business_client.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def assets():
    return {kind: {"name": kind + ".safetensors", "sha256": "a" * 64} for kind in C.ASSET_FOLDERS}


def refs(count=2):
    return [{"name": f"prepared/ref-{i}.png", "width": 480 if i % 2 == 0 else 832,
             "height": 832 if i % 2 == 0 else 480} for i in range(count)]


def event(kind, **data):
    return {"type": kind, "data": {"prompt_id": "synthetic-job", **data}}


def history(graph):
    return {"status": {"completed": True, "status_str": "success", "messages": []},
            "prompt": [0, "synthetic-job", graph],
            "outputs": {C.SAVE_ID: {"images": [{"filename": "case_00001_.mp4", "subfolder": "unique",
                                                     "type": "output"}]}}}


class Tests(unittest.TestCase):
    def test_graph_2_and_6_refs_preserves_order_and_contract(self):
        for count in (2, 6):
            graph = C.build_graph("synthetic prompt", refs(count), 1234, assets(), "unique/case")
            inputs = graph["6"]["inputs"]
            self.assertEqual((inputs["width"], inputs["height"], inputs["length"], inputs["ref_image_size"]), (480, 832, 362, "max"))
            for i in range(count):
                self.assertEqual(inputs[f"ref_images.ref_image_{i}"], [str(100+i), 0])
                self.assertEqual(graph[str(100+i)]["inputs"]["image"], refs(count)[i]["name"])
            self.assertEqual(graph["9"]["inputs"], {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0})
            self.assertEqual(graph["8"]["inputs"]["sampler_name"], "euler")
            self.assertEqual(graph["7"]["class_type"], "BasicGuider")
            self.assertEqual(graph["14"]["inputs"]["format.codec.encoding"], "auto")

    def test_seeds_change_noise_dependency_not_prompt_or_references(self):
        first = C.build_graph("constant", refs(), 1234, assets(), "unique/case")
        second = C.build_graph("constant", refs(), 5678, assets(), "unique/case")
        self.assertNotEqual(first["15"], second["15"])
        first.pop("15"); second.pop("15")
        self.assertEqual(first, second)

    def test_floor32_dimensions_match_fastvideo_and_do_not_upscale(self):
        for source, wanted in [((864,1536),(480,832)), ((1536,864),(832,480)),
                               ((480,832),(480,832)), ((455,789),(448,768)), ((33,65),(32,32))]:
            self.assertEqual(C.cap480_size(*source), wanted)
            self.assertTrue(all(a <= b for a,b in zip(wanted,source)))
        with self.assertRaises(C.ContractError): C.cap480_size(20,40)
        with self.assertRaises(C.ContractError): C.cap480_size(100,900)

    def test_traversal_external_origins_and_bad_reference_shapes_rejected(self):
        for name in ("../x", "/tmp/x", "a/../x", "a\\x", "a//x"):
            with self.assertRaises(C.ContractError): C.relative_name(name)
        for server in ("https://example.com", "http://user:pass@localhost:8188", "http://localhost:8188/x"):
            with self.assertRaises(C.ContractError): C.normalize_server(server)
        with self.assertRaises(C.ContractError): C.build_graph("x", [{"name":"x.png","width":864,"height":1536}], 1, assets(), "x")

    def test_sampler_cached_or_missing_executing_is_not_inference(self):
        good = history({})
        executing = event("executing", node=C.SAMPLER_ID)
        self.assertTrue(C.event_evidence([executing], good, "synthetic-job", C.SAMPLER_ID)["sampling_execution_verified"])
        for events in ([executing,event("execution_cached",nodes=[C.SAMPLER_ID])], [event("execution_cached",nodes=["1"])],
                       [event("executing",node=C.SAMPLER_ID,prompt_id="other")]):
            self.assertFalse(C.event_evidence(events, good, "synthetic-job", C.SAMPLER_ID)["sampling_execution_verified"])
        good["status"]["messages"] = [["execution_error", {"exception_message":"do not disclose"}]]
        self.assertFalse(C.event_evidence([executing], good, "synthetic-job", C.SAMPLER_ID)["sampling_execution_verified"])

    def test_output_prefix_and_symlink_are_confined(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"unique").mkdir()
            (root/"unique/case_00001_.mp4").write_bytes(b"synthetic non-media bytes")
            self.assertEqual(C.output_file(history({}), C.SAVE_ID, root, "unique/case"), (root/"unique/case_00001_.mp4").resolve())
            with self.assertRaises(C.ContractError): C.output_file(history({}), C.SAVE_ID, root, "other/case")
            (root/"unique/case_00001_.mp4").unlink()
            (root/"unique/case_00001_.mp4").symlink_to("/etc/hosts")
            with self.assertRaises(C.ContractError): C.output_file(history({}), C.SAVE_ID, root, "unique/case")

    def test_transport_success_requires_history_graph_and_delivered_hash(self):
        graph = C.build_graph("PRIVATE_SYNTHETIC_TEXT", refs(),1234,assets(),"unique/case")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/"server/unique").mkdir(parents=True)
            content = b"fake MP4 transport bytes, never decoded"
            (root/"server/unique/case_00001_.mp4").write_bytes(content)
            messages = [event("execution_cached",nodes=["1","3"]), event("executing",node=C.SAMPLER_ID),
                        event("execution_success")]
            class Socket:
                def settimeout(self, value): pass
                def recv(self): return json.dumps(messages.pop(0))
                def close(self): pass
            def transport(server,path,timeout,payload=None):
                return {"prompt_id":"synthetic-job"} if path=="/prompt" else {"synthetic-job":history(graph)}
            def validate(path,ffprobe):
                return {"valid":True,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
            result = C.run_request("http://127.0.0.1:8188",graph,root/"client",root/"server","unique/case",
                                   transport=transport,websocket_factory=lambda *a,**k:Socket(),media_validator=validate)
            self.assertTrue(result["included"])
            manifest = (root/"client/manifest.json").read_text()
            self.assertNotIn("PRIVATE_SYNTHETIC_TEXT",manifest)
            self.assertNotIn('"synthetic-job"',manifest)
            self.assertGreater(result["timings"]["client_to_media_s"],0)

    def test_execution_error_and_unknown_submit_fail_closed_no_retry(self):
        graph = C.build_graph("private",refs(),1234,assets(),"unique/case")
        for accepted in (True,False):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); calls=[]
                class Socket:
                    def settimeout(self,value):pass
                    def recv(self):return json.dumps(event("execution_error",exception_message="private prompt"))
                    def close(self):pass
                def transport(*args):
                    calls.append(args[1])
                    if accepted:return {"prompt_id":"synthetic-job"}
                    raise TimeoutError("could contain private server info")
                result=C.run_request("http://localhost:8188",graph,root/"client",root,"unique/case",
                                     transport=transport,websocket_factory=lambda *a,**k:Socket())
                self.assertFalse(result["included"])
                self.assertEqual(calls,["/prompt"])
                self.assertEqual(result["server_may_still_be_running"],not accepted)
                self.assertNotIn("private prompt",json.dumps(result))

    def test_summary_keeps_first_request_out_of_warm(self):
        rows=[{"phase":"warmup","included":True,"timings":{"client_to_media_s":100},"status":"completed"},
              {"phase":"measured","included":True,"timings":{"client_to_media_s":20},"status":"completed"},
              {"phase":"measured","included":False,"status":"failed"},
              {"phase":"measured","included":False,"status":"not_run"}]
        value=C.summarize(rows)
        self.assertEqual(value["cold_first_request_s"],100)
        self.assertEqual(value["warm"],{"n":1,"median_s":20,"min_s":20,"max_s":20})
        self.assertEqual((value["failed"],value["not_run"]),(1,1))

    def test_sealed_protocol_order_and_private_input_confinement(self):
        # Fixtures are arbitrary .bin bytes, never image files or decoded media.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); (root/"inputs").mkdir(); (root/"plans").mkdir()
            source=root/"inputs/ref.bin"; source.write_bytes(b"test reference")
            prompt="synthetic private prompt"
            ph=hashlib.sha256(prompt.encode()).hexdigest()
            requests=[{"case_id":"case-01","prompt":prompt,"seed":seed,"phase":"warmup" if i==0 else "measured",
                       "width":480,"height":832,"frames":362,"references":[{"type":"image","source":str(source)}]}
                      for i,seed in enumerate(C.SEEDS)]
            plan=root/"plans/cap.json";plan.write_text(json.dumps({"requests":requests}))
            protocol={"schema_version":1,"expected":C.EXPECTED,"cases":[{"case_id":"case-01","prompt_sha256":ph,
                       "references":[{"remote_path":str(source),"sha256":C.sha_file(source),"width":864,"height":1536}]}],
                      "jobs":[{"case_id":"case-01","policy_id":"cap480","resize_policy":"cap","max_short_edge":480,
                               "plan":"plans/cap.json","plan_sha256":C.sha_file(plan)}]}
            p=root/"protocol.json";p.write_text(json.dumps(protocol))
            result=C.load_case(p,C.sha_file(p),"case-01",root,root/"inputs")
            self.assertEqual(result["prompt_sha256"],ph)
            requests[1]["seed"]=9000;plan.write_text(json.dumps({"requests":requests}))
            protocol["jobs"][0]["plan_sha256"]=C.sha_file(plan);p.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(C.ContractError,"case_or_seed_mismatch"):
                C.load_case(p,C.sha_file(p),"case-01",root,root/"inputs")
            requests[1]["seed"]=1234
            requests[0]["references"][0]["source"]=str(root/"protocol.json")
            plan.write_text(json.dumps({"requests":requests}));protocol["jobs"][0]["plan_sha256"]=C.sha_file(plan)
            p.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(C.ContractError,"path_outside_allowed_root"):
                C.load_case(p,C.sha_file(p),"case-01",root,root/"inputs")

    def test_service_receipt_pin_paths_and_metadata_guard(self):
        state={"status":"READY","comfy_commit":C.COMFY_PIN,"disable_metadata":True,"gpu":"1", "port":8188,
               "input_dir":"/project/input","output_dir":"/project/output","started_at":1,"pid":100,"owner_token":"synthetic"}
        output=C.verify_service_receipt(state,"http://127.0.0.1:8188","/project/input","/project/output")
        self.assertEqual(output["gpu"],1)
        self.assertNotIn("owner_token",output)
        for field,value in (("disable_metadata",False),("comfy_commit","f"*40),("output_dir","/elsewhere")):
            with self.assertRaises(C.ContractError):
                C.verify_service_receipt(dict(state,**{field:value}),"http://127.0.0.1:8188","/project/input","/project/output")


if __name__ == "__main__":
    unittest.main()
