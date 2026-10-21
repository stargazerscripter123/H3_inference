"""Scoped Ray health compatibility for FastVideo cdda1b75.

The pinned upstream health property assumes multiprocessing Worker.proc.
Ray ActorHandles instead expose _actor_id; query each exact actor in GCS.
This checks process liveness (like MP is_alive), not numerical GPU correctness.
GCS inspection is independent of the actor method queue, so an in-flight long
generation does not make a healthy actor time out behind its own work.
"""
from __future__ import annotations

import inspect


def install_ray_health_adapter(engine_class=None, actor_lookup=None) -> None:
    if engine_class is None:
        from fastvideo.entrypoints.openai.serving_engine import OpenAIServingEngine
        engine_class = OpenAIServingEngine
    if getattr(engine_class, "_omni_ray_health_installed", False):
        return
    original = engine_class.healthy.fget
    if engine_class.__module__.startswith("fastvideo.") and "worker.proc.is_alive()" not in inspect.getsource(original):
        raise RuntimeError("Upstream health implementation changed; review compatibility adapter")
    if actor_lookup is None:
        # Pinned Ray 2.49.1 source: actors(hex_id) -> {State: ALIVE/DEAD/...}.
        # This direct GCS accessor needs no dashboard or additional state server.
        from ray._private.state import actors
        actor_lookup = actors

    def healthy(self):
        executor = getattr(self._generator, "executor", None)
        args = getattr(executor, "fastvideo_args", None)
        if getattr(args, "distributed_executor_backend", None) != "ray":
            return original(self)
        self._omni_ray_health_reason = None
        if self._closed or self._unhealthy_reason is not None:
            return False
        workers = getattr(executor, "workers", None)
        if not workers or len(workers) != args.num_gpus:
            self._omni_ray_health_reason = "Ray worker count does not match requested GPU count"
            return False
        try:
            for worker in workers:
                record = actor_lookup(worker._actor_id.hex())
                if record.get("State") != "ALIVE":
                    self._omni_ray_health_reason = f"Ray actor is {record.get('State', 'UNKNOWN')}"
                    return False
        except Exception as error:
            # A failed control-plane query is not proof of actor health.
            self._omni_ray_health_reason = f"Ray liveness query failed: {type(error).__name__}"
            return False
        return True

    original_reason = engine_class.unhealthy_reason.fget

    def unhealthy_reason(self):
        if getattr(self, "_omni_ray_health_reason", None):
            return self._omni_ray_health_reason
        return original_reason(self)

    engine_class.healthy = property(healthy)
    engine_class.unhealthy_reason = property(unhealthy_reason)
    engine_class._omni_ray_health_installed = True
