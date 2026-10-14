"""The launch preflights check what the launch will actually do.

Each preflight answers a question about the launch that follows it, so it has
to ask with the launch's own inputs: the targets' cache (not the control
machine's), the launch's own executors (not one re-resolved from the serve
overrides), and the exact model set distribution would move.
"""

from types import SimpleNamespace

import pytest

from sparkrun.core import launcher
from sparkrun.core.recipe import Recipe, RecipeError


class _Stop(Exception):
    """Ends a launch once the code under test has been reached."""


class _StubExecutor:
    needs_image = True

    def __init__(self, cli_overrides=None):
        self.cli_overrides = dict(cli_overrides or {})
        self.mount_probes = []
        self.missing = {}

    def bind_host_executors(self, executors):
        _StubExecutor.bound = dict(executors)

    def bind_image_references(self, references):
        pass

    def bind_mount_sources(self):
        return list(self.cli_overrides.get("volumes", []))

    def verify_mount_sources(self, paths, hosts, *, ssh_kwargs=None):
        self.mount_probes.append((list(paths), list(hosts)))
        return {h: list(paths) for h in hosts if self.missing}

    def verify_command_passthrough(self, image, hosts, *, ssh_kwargs=None):
        return None


@pytest.fixture
def launch(monkeypatch, tmp_path):
    from test_launcher import _StubRuntime

    class Config(dict):
        hf_cache_dir = tmp_path / "control-hf"
        cache_dir = tmp_path / "cache"
        missing_mount_source_policy = "fail"
        readiness_port_timeout_s = 240
        readiness_health_timeout_s = 600

        def get_registry_manager(self):
            return None

    class Runtime(_StubRuntime):
        def resolve_container(self, recipe, **kwargs):
            return recipe.container

    monkeypatch.setattr(_StubExecutor, "bound", {}, raising=False)

    def resolve_executor(**kw):
        return _StubExecutor(kw.get("cli_overrides"))

    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda *a, **kw: SimpleNamespace(mode="local"))
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_preflight_test")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: "/home/remote/.cache/huggingface")
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", resolve_executor)

    def run(**kwargs):
        recipe = Recipe.from_dict(
            {"name": "preflight", "runtime": "vllm-distributed", "model": "org/model", "container": "img:1", "defaults": {"port": 8000}}
        )
        launcher.launch_inference(
            recipe=recipe, runtime=Runtime(), host_list=["h1", "h2"], overrides={}, config=Config(), sync_tuning=False, **kwargs
        )

    run.config_hf = tmp_path / "control-hf"
    return run


def test_mount_preflight_probes_cli_executor_volumes(monkeypatch, launch):
    """CLI ``executor_config`` volumes are what the launch binds, so they are what gets checked."""

    def missing_everywhere(self, paths, hosts, *, ssh_kwargs=None):
        return {h: list(paths) for h in hosts}

    monkeypatch.setattr(_StubExecutor, "verify_mount_sources", missing_everywhere)
    with pytest.raises(RecipeError, match="/cli/vol"):
        launch(executor_config={"volumes": ["/cli/vol"]})


def test_mount_preflight_uses_the_launch_executors(monkeypatch, launch):
    seen = {}

    def capture(*a, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(launcher, "_verify_mount_sources", capture)
    with pytest.raises(_Stop):
        launch(executor_config={"executor": "local"})
    # The probe gets exactly the executors the launch runs with — no second resolution.
    assert seen["host_executors"] == _StubExecutor.bound
    assert all(ex.cli_overrides == {"executor": "local"} for ex in seen["host_executors"].values())


def test_offline_preflight_reads_target_cache_and_distribution_model_set(monkeypatch, launch):
    preflight = {}
    distribution = {}

    monkeypatch.setattr(launcher, "_offline_preflight", lambda *a, **kw: preflight.update(kw))

    def distribute(*a, **kw):
        distribution.update(kw, cache_dir=a[3])
        raise _Stop

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    with pytest.raises(_Stop):
        launch(offline=True, skip_model_fan_out=True)

    # Host-side checks look where distribution writes: the targets' cache.
    assert preflight["cache_dir"] == "/home/remote/.cache/huggingface" == distribution["cache_dir"]
    # The control-side check looks where distribution copies from.
    assert preflight["local_cache_dir"] == str(launch.config_hf) == distribution["local_cache_dir"]
    # And it checks exactly the model set distribution would move.
    assert preflight["skip_model"] is distribution["skip_model"] is False
    assert preflight["skip_images"] is distribution["skip_container"] is False
    assert preflight["shared_model_cache"] is distribution["prefs"].skip_fan_out is True


def test_entrypoint_probe_uses_each_hosts_launch_executor(monkeypatch, launch):
    probes = []
    monkeypatch.setattr(launcher, "_verify_image_command_passthrough", lambda recipe, image, hosts, *a, **kw: probes.append((hosts[0], kw)))

    def distribute(*a, **kw):
        kw["after_container_sync"]()
        raise _Stop

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    with pytest.raises(_Stop):
        launch(executor_config={"entrypoint": ""})

    assert probes, "the entrypoint probe must run"
    for host, kw in probes:
        assert kw["executor"] is _StubExecutor.bound[host]
        assert kw["executor"].cli_overrides == {"entrypoint": ""}


# --------------------------------------------------------------------------
# Unit level: a supplied executor is used, never re-resolved
# --------------------------------------------------------------------------


def _no_resolution(monkeypatch):
    def boom(**kw):
        raise AssertionError("must not re-resolve an executor the launch supplied")

    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", boom)


def test_entrypoint_preflight_uses_supplied_executor(monkeypatch):
    from sparkrun.containers.entrypoint import VERDICT_CONSUMES, EntrypointProbe

    _no_resolution(monkeypatch)
    executor = _StubExecutor()
    executor.verify_command_passthrough = lambda image, hosts, **kw: EntrypointProbe(
        image=image, host=hosts[0], verdict=VERDICT_CONSUMES, entrypoint='["vllm","serve"]'
    )
    with pytest.raises(RecipeError, match="ENTRYPOINT"):
        launcher._verify_image_command_passthrough(
            None,
            "img:1",
            ["h1"],
            {},
            runtime=None,
            cluster=None,
            config=None,
            executor_config=None,
            rootless=True,
            auto_user=True,
            host_hardware=None,
            v=None,
            executor=executor,
        )


def _verify(host_executors, recipe=None):
    launcher._verify_mount_sources(
        recipe or SimpleNamespace(model="org/model", cluster_config=None),
        list(host_executors),
        {},
        runtime=None,
        cluster=None,
        config=None,
        overrides={},
        host_executors=host_executors,
    )


def test_mount_preflight_shares_one_probe_across_matching_executors(monkeypatch):
    _no_resolution(monkeypatch)
    a, b = _StubExecutor({"volumes": ["/v"]}), _StubExecutor({"volumes": ["/v"]})
    _verify({"h1": a, "h2": b})
    assert a.mount_probes == [(["/v"], ["h1", "h2"])] and b.mount_probes == []


def test_mount_preflight_probes_each_hosts_own_volumes(monkeypatch):
    _no_resolution(monkeypatch)
    a, b = _StubExecutor({"volumes": ["/a"]}), _StubExecutor({"volumes": ["/b"]})
    b.missing = True
    with pytest.raises(RecipeError, match="h2: /b"):
        _verify({"h1": a, "h2": b})
    assert a.mount_probes == [(["/a"], ["h1"])] and b.mount_probes == [(["/b"], ["h2"])]
