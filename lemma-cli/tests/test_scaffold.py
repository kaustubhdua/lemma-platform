from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lemma_cli.cli_app.pod_bundle import load_resource_payload, loads_jsonc, strip_jsonc
from lemma_cli.cli_app.scaffold import (
    ScaffoldError,
    find_bundle_root,
    grant_in_bundle,
    init_pod,
    init_resource,
    merge_grants,
    parse_grant_spec,
    pascalify,
    resolve_root,
    resource_example,
    slugify,
    snakeify,
    templatize_bundle,
)


# --------------------------------------------------------------------------- #
# JSONC
# --------------------------------------------------------------------------- #
def test_strip_jsonc_line_and_block_comments():
    text = """{
  // a line comment
  "a": 1,  // trailing comment
  "b": 2   /* block */
}"""
    assert loads_jsonc(text) == {"a": 1, "b": 2}


def test_strip_jsonc_preserves_slashes_in_strings():
    text = '{ "url": "http://example.com/a//b", "note": "1/2" }'
    assert loads_jsonc(text) == {"url": "http://example.com/a//b", "note": "1/2"}


def test_strip_jsonc_preserves_line_numbers():
    # Comment replaced in place so an error on line 3 still reports line 3.
    text = "{\n  // comment\n  bad\n}"
    stripped = strip_jsonc(text)
    assert stripped.count("\n") == text.count("\n")


def test_strip_jsonc_block_comment_with_star_and_quote():
    text = '{ "a": /* keep "this" star * slash */ 5 }'
    assert loads_jsonc(text) == {"a": 5}


# --------------------------------------------------------------------------- #
# name helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Lead Qualifier!!", "lead-qualifier"),
        ("triage_agent", "triage-agent"),
        ("A B C", "a-b-c"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("Score The Lead", "score_the_lead"), ("my-func", "my_func"), ("X", "x")],
)
def test_snakeify(raw, expected):
    assert snakeify(raw) == expected


def test_pascalify():
    assert pascalify("score the lead") == "ScoreTheLead"
    assert pascalify("my-func") == "MyFunc"


def test_name_helpers_reject_empty():
    with pytest.raises(ScaffoldError):
        slugify("!!!")
    with pytest.raises(ScaffoldError):
        snakeify("###")


# --------------------------------------------------------------------------- #
# init_pod
# --------------------------------------------------------------------------- #
def test_init_pod_creates_parsable_bundle(tmp_path: Path):
    result = init_pod(tmp_path / "demo", "Demo Ops")
    assert result.name == "demo-ops"
    names = {p.name for p in result.files}
    assert {
        "pod.json",
        "items.json",
        "hello.json",
        "instruction.md",
        "README.md",
        "AGENTS.md",
    } <= names

    # Every scaffolded JSON parses as JSONC and matches the folder==name rule.
    root = tmp_path / "demo"
    assert load_resource_payload(root / "tables" / "items", "items")["name"] == "items"
    agent = load_resource_payload(root / "agents" / "hello", "hello")
    assert agent["visibility"] == "POD"
    # the starter agent is granted on the items table
    grants = agent["permissions"]["grants"]
    assert any(g["resource_name"] == "items" for g in grants)
    # instruction.md resolved via $file ref into a string
    assert isinstance(agent["instruction"], str) and agent["instruction"].strip()


def test_init_pod_refuses_existing_without_force(tmp_path: Path):
    init_pod(tmp_path / "demo", "demo")
    with pytest.raises(ScaffoldError):
        init_pod(tmp_path / "demo", "demo")
    # force overwrites
    init_pod(tmp_path / "demo", "demo", force=True)


def test_init_pod_force_overwrites_partial_bundle_and_preserves_unrelated_files(tmp_path: Path):
    root = tmp_path / "partial"
    root.mkdir()
    (root / "pod.json").write_text("stale manifest", encoding="utf-8")
    (root / "README.md").write_text("stale readme", encoding="utf-8")
    unrelated = root / "notes.txt"
    unrelated.write_text("keep this file", encoding="utf-8")

    result = init_pod(root, "Fresh Pod", force=True)

    assert result.name == "fresh-pod"
    assert '"name": "fresh-pod"' in (root / "pod.json").read_text(encoding="utf-8")
    assert "# fresh-pod" in (root / "README.md").read_text(encoding="utf-8")
    assert (root / "tables" / "items" / "items.json").is_file()
    assert (root / "agents" / "hello" / "hello.json").is_file()
    assert unrelated.read_text(encoding="utf-8") == "keep this file"


# --------------------------------------------------------------------------- #
# init_resource
# --------------------------------------------------------------------------- #
def test_init_function_headers_match_folder(tmp_path: Path):
    result = init_resource("function", "Score The Lead", root=tmp_path)
    assert result.name == "score_the_lead"
    code = (tmp_path / "functions" / "score_the_lead" / "code.py").read_text()
    ast.parse(code)  # valid python
    header_lines = [ln for ln in code.splitlines() if ln.startswith("#")][:3]
    assert "#function_name: score_the_lead" in header_lines
    assert "#input_type_name: ScoreTheLeadInput" in header_lines


def test_init_table_shared_flag(tmp_path: Path):
    init_resource("table", "widgets", root=tmp_path, shared=True)
    shared = loads_jsonc((tmp_path / "tables" / "widgets" / "widgets.json").read_text())
    assert shared["enable_rls"] is False

    init_resource("table", "secrets", root=tmp_path)  # default = RLS on
    private = loads_jsonc(
        (tmp_path / "tables" / "secrets" / "secrets.json").read_text()
    )
    assert private["enable_rls"] is True

def test_init_every_resource_type_parses(tmp_path: Path):
    init_resource("table", "items", root=tmp_path)
    init_resource("function", "do_it", root=tmp_path)
    init_resource("agent", "helper", root=tmp_path)
    init_resource("workflow", "flow", root=tmp_path)
    init_resource("schedule", "nightly", root=tmp_path)
    init_resource("surface", "slack", root=tmp_path)
    for rdir, name in [
        ("tables/items", "items"),
        ("functions/do_it", "do_it"),
        ("agents/helper", "helper"),
        ("workflows/flow", "flow"),
        ("schedules/nightly", "nightly"),
        ("surfaces/slack", "slack"),
    ]:
        payload = load_resource_payload(tmp_path / rdir, name)
        assert payload  # parsed, non-empty
        # visibility is a first-class bundle field on the resources that carry it
        if name not in {"slack"}:
            assert payload.get("visibility") == "POD"


def test_init_workflow_graph_is_valid_shape(tmp_path: Path):
    init_resource("workflow", "flow", root=tmp_path)
    wf = load_resource_payload(tmp_path / "workflows" / "flow", "flow")
    node_ids = {n["id"] for n in wf["nodes"]}
    assert {"intake", "end"} <= node_ids
    for edge in wf["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids


def test_init_unknown_resource_type(tmp_path: Path):
    with pytest.raises(ScaffoldError):
        init_resource("gadget", "x", root=tmp_path)


# --------------------------------------------------------------------------- #
# bundle root resolution
# --------------------------------------------------------------------------- #
def test_find_bundle_root_walks_up(tmp_path: Path):
    init_pod(tmp_path / "demo", "demo")
    nested = tmp_path / "demo" / "agents" / "assistant"
    assert find_bundle_root(nested) == (tmp_path / "demo").resolve()


def test_find_bundle_root_none_when_absent(tmp_path: Path):
    assert find_bundle_root(tmp_path) is None


def test_resolve_root_prefers_explicit(tmp_path: Path):
    explicit = tmp_path / "x"
    assert resolve_root(explicit) == explicit


# --------------------------------------------------------------------------- #
# grants (X7)
# --------------------------------------------------------------------------- #
def test_parse_grant_spec_table_inferred():
    g = parse_grant_spec("tickets:read,write")
    assert g["resource_type"] == "datastore_table"
    assert g["resource_name"] == "tickets"
    assert g["permission_ids"] == [
        "datastore.table.read",
        "datastore.record.read",
        "datastore.record.write",
    ]


def test_parse_grant_spec_folder_inferred_by_slash():
    g = parse_grant_spec("/knowledge:read")
    assert g["resource_type"] == "folder"
    assert g["permission_ids"] == ["folder.read"]


def test_parse_grant_spec_explicit_app_and_raw_perm():
    g = parse_grant_spec("app:gmail:use")
    assert g["resource_type"] == "connector"
    assert g["permission_ids"] == ["connector.use"]
    raw = parse_grant_spec("table:tickets:datastore.record.write")
    assert raw["permission_ids"] == ["datastore.record.write"]


def test_parse_grant_spec_table_delete_is_structural():
    # `delete` means the structural (destructive) table delete; deleting rows
    # is `delete-rows`, which is datastore.record.delete (also destructive).
    g = parse_grant_spec("tickets:delete")
    assert g["permission_ids"] == ["datastore.table.delete"]


def test_parse_grant_spec_delete_rows_is_its_own_permission():
    # Row deletes stopped riding on datastore.record.write: writing a row is
    # not deleting every row.
    g = parse_grant_spec("tickets:write,delete-rows")
    assert g["permission_ids"] == ["datastore.record.write", "datastore.record.delete"]


def test_parse_grant_spec_function_and_agent_tools():
    # Functions/agents can be granted as tools; execute is self-sufficient.
    fn = parse_grant_spec("function:cycle_velocity:execute")
    assert fn["resource_type"] == "function"
    assert fn["resource_name"] == "cycle_velocity"
    assert fn["permission_ids"] == ["function.execute"]

    ag = parse_grant_spec("agent:triage:execute")
    assert ag["resource_type"] == "agent"
    assert ag["permission_ids"] == ["agent.execute"]

    # One token per verb: `use` is for connector apps, not functions/agents.
    with pytest.raises(ScaffoldError):
        parse_grant_spec("function:cycle_velocity:use")


def test_parse_grant_spec_rejects_bad_input():
    with pytest.raises(ScaffoldError):
        parse_grant_spec("tickets")  # no perms
    with pytest.raises(ScaffoldError):
        parse_grant_spec("tickets:bogus")  # unknown perm token


def test_merge_grants_unions_permission_ids():
    existing = [
        {
            "resource_type": "datastore_table",
            "resource_name": "t",
            "permission_ids": ["a"],
        }
    ]
    new = [
        {
            "resource_type": "datastore_table",
            "resource_name": "t",
            "permission_ids": ["a", "b"],
        }
    ]
    merged = merge_grants(existing, new)
    assert len(merged) == 1
    assert merged[0]["permission_ids"] == ["a", "b"]


def test_grant_in_bundle_merges_into_file(tmp_path: Path):
    init_resource("agent", "triage", root=tmp_path)
    path, perms = grant_in_bundle(
        "agent", "triage", ["tickets:read", "/kb:read"], root=tmp_path
    )
    names = {g["resource_name"] for g in perms["grants"]}
    assert {"tickets", "/kb"} <= names
    # second call unions, not duplicates
    _, perms2 = grant_in_bundle("agent", "triage", ["tickets:write"], root=tmp_path)
    tickets = [g for g in perms2["grants"] if g["resource_name"] == "tickets"][0]
    assert "datastore.record.write" in tickets["permission_ids"]
    assert sum(1 for g in perms2["grants"] if g["resource_name"] == "tickets") == 1


def test_grant_in_bundle_requires_scaffold(tmp_path: Path):
    with pytest.raises(ScaffoldError):
        grant_in_bundle("agent", "missing", ["t:read"], root=tmp_path)


# --------------------------------------------------------------------------- #
# runtime pin (AG2) + schema (X10) + templatize (P3)
# --------------------------------------------------------------------------- #
def test_init_agent_with_runtime(tmp_path: Path):
    init_resource("agent", "pinned", root=tmp_path, runtime="profile-123")
    from lemma_cli.cli_app.pod_bundle import loads_jsonc

    data = loads_jsonc((tmp_path / "agents" / "pinned" / "pinned.json").read_text())
    assert data["agent_runtime"] == {"profile_id": "profile-123"}


def test_resource_example_parses_for_each_type():