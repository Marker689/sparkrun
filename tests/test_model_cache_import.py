"""Origin-resolved imports require consent and preserve offline validation semantics."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from sparkrun.api import model_cache as api
from sparkrun.cli import _model_cache as cli
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest, ModelInventoryUnavailable
from sparkrun.models.manifest import manifest_path
from test_model_artifacts import artifact, cache, SHA


@pytest.fixture
def origin(monkeypatch):
    manifest = artifact()
    inventory = Mock(return_value={"revision": SHA, "files": [asdict(f) for f in manifest.files]})
    monkeypatch.setattr(api, "fetch_inventory", inventory)
    monkeypatch.setattr("sparkrun.core.config.resolve_hf_token", lambda: "server-token")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", Mock(side_effect=AssertionError("unexpected payload download")))
    return inventory


@pytest.mark.parametrize("options", [{}, {"fetch_manifest": False}, {"offline": True}, {"offline": True, "fetch_manifest": True}])
def test_missing_manifest_requires_consent_and_online_mode(tmp_path, origin, options):
    with pytest.raises(ModelArtifactError, match="[Mm]anifest"):
        api.import_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), **options)
    origin.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("token", [None, "explicit-token"])
def test_origin_import_pins_revision_and_keeps_payload_and_usage_unchanged(tmp_path, origin, token):
    request = ModelArtifactRequest("org/model")
    snapshot = cache(tmp_path, artifact())
    before = {p.name: p.read_bytes() for p in snapshot.iterdir()}
    reports = api.import_model(request, cache_dir=str(tmp_path), fetch_manifest=True, token=token, checksums=True)
    assert reports[0].complete and reports[0].level == "checksum"
    origin.assert_called_once_with(request, token=token or "server-token", cache=None)
    saved = ModelArtifactManifest.from_json(manifest_path(tmp_path, request).read_text())
    assert saved.revision == SHA and saved.provenance == "huggingface-model-info"
    assert {p.name: p.read_bytes() for p in snapshot.iterdir()} == before
    assert not (tmp_path / ".sparkrun-model-artifacts/usage").exists()


def test_origin_resolution_occurs_once_for_all_target_hosts(tmp_path, monkeypatch, origin):
    io = Mock()
    io.root.side_effect = [str(tmp_path / "node1"), str(tmp_path / "node2")]
    monkeypatch.setattr(api, "ModelHostIO", Mock(return_value=Mock(__enter__=Mock(return_value=io), __exit__=Mock(return_value=False))))
    request = ModelArtifactRequest("org/model", "main")
    api.import_model(request, cache_dir="/remote/hf", hosts=("node1", "node2"), fetch_manifest=True)
    origin.assert_called_once()
    assert [call.args[0] for call in io.observe.call_args_list] == ["node1", "node2"]
    assert all(call.args[1].revision == SHA for call in io.observe.call_args_list)
    assert io.observe.call_args_list[0].args[1] is io.observe.call_args_list[1].args[1]
    assert len(io.publish.call_args_list) == 2
    assert io.publish.call_args_list[0].args[-1] == io.publish.call_args_list[1].args[-1]


@pytest.mark.parametrize("offline", [False, True])
def test_supplied_manifest_bypasses_resolution_and_confirmation(tmp_path, origin, offline):
    request = ModelArtifactRequest("org/model", SHA)
    confirm = Mock(side_effect=AssertionError("supplied manifests do not prompt"))
    cache(tmp_path, artifact())
    assert api.import_model(request, artifact(), cache_dir=str(tmp_path), fetch_manifest=True, offline=offline, confirm_manifest=confirm)[
        0
    ].complete
    origin.assert_not_called()
    confirm.assert_not_called()


def test_declining_resolved_manifest_preserves_saved_manifest_and_does_not_open_hosts(tmp_path, monkeypatch, origin):
    request = ModelArtifactRequest("org/model")
    existing = replace(artifact(), revision="b" * 40)
    api.import_manifest(request, existing, cache_dir=str(tmp_path))
    host_io = Mock(side_effect=AssertionError("target IO before consent"))
    monkeypatch.setattr(api, "ModelHostIO", host_io)

    def decline(manifest):
        assert manifest.revision == SHA
        assert manifest_path(tmp_path, request).read_text() == existing.to_json()
        return False

    with pytest.raises(ModelArtifactError, match="cancelled"):
        api.import_model(request, cache_dir=str(tmp_path), fetch_manifest=True, confirm_manifest=decline)
    host_io.assert_not_called()
    assert manifest_path(tmp_path, request).read_text() == existing.to_json()


def test_origin_failure_never_infers_inventory_from_local_files(tmp_path, origin):
    request = ModelArtifactRequest("org/model", SHA)
    cache(tmp_path, artifact())
    origin.side_effect = ModelInventoryUnavailable("private model inventory unavailable")
    with pytest.raises(ModelInventoryUnavailable, match="supply --manifest.*HF credentials"):
        api.import_model(request, cache_dir=str(tmp_path), fetch_manifest=True)
    assert not manifest_path(tmp_path, request).exists()


@pytest.mark.parametrize("mismatch", [{"repo": "org/other"}, {"revision": "b" * 40}, {"endpoint": "https://mirror.example"}])
def test_supplied_manifest_mismatch_is_not_replaced_by_origin(tmp_path, origin, mismatch):
    with pytest.raises(ModelArtifactError):
        api.import_model(
            ModelArtifactRequest("org/model", SHA), replace(artifact(), **mismatch), cache_dir=str(tmp_path), fetch_manifest=True
        )
    origin.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_resolved_commit_must_match_requested_commit_before_confirmation(tmp_path, origin):
    origin.return_value["revision"] = "b" * 40
    confirm = Mock()
    with pytest.raises(ModelArtifactError, match="different revision"):
        api.import_model(ModelArtifactRequest("org/model", SHA), cache_dir=str(tmp_path), fetch_manifest=True, confirm_manifest=confirm)
    confirm.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_origin_import_uses_shared_gguf_selection(tmp_path, origin):
    request = ModelArtifactRequest("org/model:Q4_K_M", projector="mmproj.gguf")
    origin.return_value["files"] = [{"path": p, "size": 10} for p in ("model-Q4_K_M.gguf", "model-Q8_0.gguf", "mmproj.gguf")]
    result = api.import_model(request, cache_dir=str(tmp_path), fetch_manifest=True)
    saved = ModelArtifactManifest.from_json(manifest_path(tmp_path, request).read_text())
    assert saved.weight_path == "model-Q4_K_M.gguf" and saved.projector_path == "mmproj.gguf"
    assert {f.path for f in saved.files} == {"model-Q4_K_M.gguf", "mmproj.gguf"}
    assert result[0].status == "incomplete"


@pytest.fixture
def invoke(tmp_path, monkeypatch):
    config = SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=None, default_hosts=None)
    context = SimpleNamespace(config=config, cluster_manager=Mock(get_default=Mock(return_value=None)))
    monkeypatch.setattr(cli, "_get_context", lambda ctx: context)
    monkeypatch.setattr(cli, "_interactive", lambda: False)

    def run(*args, input=None):
        return CliRunner().invoke(cli.model_cache, ["import", "org/model", "--cache-dir", str(tmp_path), *args], input=input)

    return run


@pytest.mark.parametrize("options", [(), ("--json",), ("--offline",), ("--offline", "--yes")])
def test_cli_missing_manifest_refuses_unattended_or_offline_origin_lookup(tmp_path, origin, invoke, options):
    result = invoke(*options)
    assert result.exit_code == 1, result.output
    assert "--manifest" in result.output
    origin.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("answer", ["n\n", "\n", "y\n"])
def test_cli_interactive_confirmation_previews_resolved_identity(tmp_path, monkeypatch, origin, invoke, answer):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    cache(tmp_path, artifact())
    result = invoke(input=answer)
    assert "https://huggingface.co" in result.output and SHA in result.output
    assert "Selected files: 3" in result.output and "Targets: localhost" in result.output
    assert "Save this manifest" in result.output
    accepted = answer == "y\n"
    assert result.exit_code == (0 if accepted else 1), result.output
    assert manifest_path(tmp_path, ModelArtifactRequest("org/model")).exists() == accepted
    origin.assert_called_once()


def test_cli_json_requires_yes_even_with_terminal(tmp_path, monkeypatch, origin, invoke):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    result = invoke("--json", input="y\n")
    assert result.exit_code == 1 and "--yes" in result.output
    origin.assert_not_called()


@pytest.mark.parametrize("complete", [False, True])
def test_cli_yes_keeps_json_clean_and_incomplete_import_nonzero(tmp_path, origin, invoke, complete):
    request = ModelArtifactRequest("org/model")
    if complete:
        cache(tmp_path, artifact())
    result = invoke("--yes", "--json")
    assert result.exit_code == (0 if complete else 1), result.output
    reports = json.loads(result.stdout)
    assert reports[0]["status"] == ("complete" if complete else "incomplete")
    assert manifest_path(tmp_path, request).exists()
    assert SHA in result.stderr and "https://huggingface.co" in result.stderr
    assert "Save this manifest" not in result.stderr


@pytest.mark.parametrize("flags", [(), ("--offline",), ("--yes",)])
def test_cli_file_import_remains_offline_without_confirmation(tmp_path, origin, invoke, flags):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(artifact().to_json())
    cache(tmp_path, artifact())
    result = invoke("--manifest", str(manifest_file), "--json", *flags)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["status"] == "complete"
    assert not result.stderr
    origin.assert_not_called()


@pytest.mark.parametrize("contents", ["broken json", "{}", replace(artifact(), repo="org/other").to_json()])
def test_cli_invalid_file_never_falls_back_to_origin(tmp_path, origin, invoke, contents):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(contents)
    result = invoke("--manifest", str(manifest_file), "--yes")
    assert result.exit_code == 1, result.output
    origin.assert_not_called()
    assert not manifest_path(tmp_path, ModelArtifactRequest("org/model")).exists()


def test_cli_origin_failure_reports_remedy_without_saving(tmp_path, origin, invoke):
    origin.side_effect = ModelInventoryUnavailable("inventory unavailable")
    result = invoke("--yes")
    assert result.exit_code == 1, result.output
    assert "--manifest" in result.output and "HF credentials" in result.output
    assert not manifest_path(tmp_path, ModelArtifactRequest("org/model")).exists()


@pytest.mark.parametrize("fetch_manifest,offline", [(False, False), (True, True), (True, False)])
def test_daemon_import_requires_explicit_origin_consent(tmp_path, monkeypatch, origin, fetch_manifest, offline):
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.daemon.ops import check_request, register_core_ops

    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: {})
    cache(tmp_path, artifact())
    register_core_ops()
    op = get_daemon_op("model_cache.import")
    arguments = {"model": "org/model", "cache_dir": str(tmp_path), "offline": offline}
    if fetch_manifest:
        arguments["fetch_manifest"] = True
    arguments = check_request(op, arguments)
    handler = op.handler_loader()
    if not fetch_manifest or offline:
        with pytest.raises(ModelArtifactError):
            handler(arguments, SimpleNamespace(config=None))
        origin.assert_not_called()
        assert not manifest_path(tmp_path, ModelArtifactRequest("org/model")).exists()
    else:
        assert handler(arguments, SimpleNamespace(config=None))[0]["status"] == "complete"
        assert origin.call_args.kwargs["token"] == "server-token"
        assert manifest_path(tmp_path, ModelArtifactRequest("org/model")).exists()
