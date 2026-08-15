"""The graph projection pairs a SQL query with the Cypher it feeds, and the two agree only
by convention — nothing at runtime checks them against each other. Both ways they can
disagree fail *silently*:

  - a `row.x` with no matching SQL column sets that property to null, so the projection
    reports a healthy row count and the graph quietly holds empty properties;
  - a relationship whose MATCH names a label no node projection creates matches nothing,
    so the edge count is 0 and no error is raised anywhere.

These check both statically, so a typo fails `make test` instead of surfacing days later
as an empty query result. None of them needs Postgres or Neo4j.
"""
import re

import pytest

from common.config import Settings
from common.graph_sync import (MENTION_CYPHER, NODE_PROJECTIONS, PROJECTIONS,
                                REL_PROJECTIONS, SPEECH_TEXT_SQL)

IDS = [p.name for p in PROJECTIONS]


def _split_top_level(s: str) -> list[str]:
    """Split on commas that aren't inside parentheses — LAG(...) OVER (...) has both."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _output_columns(sql: str) -> set[str]:
    """Names a caller sees on the returned rows: the alias if there is one, else the bare
    column with any table qualifier dropped (`u.clip_id` is returned as `clip_id`).

    Which top-level SELECT carries the output names depends on the query shape: after a
    `WITH` the real one is the last, but in a UNION Postgres takes the names from the
    *first* branch. Getting this backwards would quietly compare against the wrong column
    list and let a genuine mismatch through."""
    flat = " ".join(sql.split())

    # SELECTs that aren't nested inside parentheses (so: not in a CTE body or subquery)
    depth, starts = 0, []
    for i, ch in enumerate(flat):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and flat[i:i + 7].upper() == "SELECT ":
            starts.append(i + 7)
    body = flat[starts[-1] if flat.upper().startswith("WITH ") else starts[0]:]

    depth, end = 0, len(body)
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and body[i:i + 6].upper() == " FROM ":
            end = i
            break
    select_list = body[:end]
    if select_list.strip().upper().startswith("DISTINCT"):
        select_list = select_list.strip()[8:]

    cols = set()
    for item in _split_top_level(select_list):
        item = item.strip()
        if not item:
            continue
        m = re.search(r"\s+AS\s+(\w+)$", item, re.IGNORECASE)
        cols.add(m.group(1) if m else item.split(".")[-1].strip())
    return cols


@pytest.mark.parametrize("proj", PROJECTIONS, ids=IDS)
def test_every_row_reference_resolves_to_a_sql_column(proj):
    referenced = set(re.findall(r"\brow\.(\w+)", proj.cypher))
    available = _output_columns(proj.sql)
    missing = referenced - available
    assert not missing, (f"{proj.name}: cypher reads {sorted(missing)}, "
                         f"sql returns {sorted(available)}")


@pytest.mark.parametrize("proj", PROJECTIONS, ids=IDS)
def test_declared_params_match_the_placeholders_in_the_sql(proj):
    """$1..$n are bound positionally from Projection.params, so a mismatch means either a
    missing argument or a threshold silently not applied."""
    placeholders = {int(n) for n in re.findall(r"\$(\d+)", proj.sql)}
    assert placeholders == set(range(1, len(proj.params) + 1)), (
        f"{proj.name}: sql uses {sorted(placeholders)}, params declares {list(proj.params)}")


@pytest.mark.parametrize("proj", PROJECTIONS, ids=IDS)
def test_declared_params_are_real_settings_fields(proj):
    for name in proj.params:
        assert name in Settings.model_fields, f"{proj.name}: no Settings field {name!r}"


def _labels_created() -> set[str]:
    """Labels a node projection actually puts on a node — via MERGE, and via `SET n:Label`
    for the secondary :Speech label that Utterance and RadioCall share."""
    created = set()
    for p in NODE_PROJECTIONS:
        created |= set(re.findall(r"MERGE\s*\(\w*:(\w+)", p.cypher))
        created |= set(re.findall(r"SET\s+\w+:(\w+)", p.cypher))
    return created


@pytest.mark.parametrize("cypher,name",
                         [(p.cypher, p.name) for p in REL_PROJECTIONS]
                         + [(c, f"MENTIONS_{k}") for k, c in MENTION_CYPHER.items()])
def test_relationship_endpoints_use_labels_some_node_projection_creates(cypher, name):
    matched = set(re.findall(r"[(,]\s*\w*:(\w+)\s*\{", cypher))
    unknown = matched - _labels_created()
    assert not unknown, f"{name}: MATCHes {sorted(unknown)}, which nothing projects"


def test_speech_label_is_applied_to_every_source_the_shared_edges_match():
    """DURING_LAP and MENTIONS are single projections only because Utterance and RadioCall
    both carry :Speech. If one stopped setting it, those edges would silently halve."""
    setters = {p.name for p in NODE_PROJECTIONS if re.search(r"SET\s+\w+:Speech", p.cypher)}
    assert setters == {"Utterance", "RadioCall"}

    for p in NODE_PROJECTIONS:
        if p.name in setters:
            assert "speech_id" in p.cypher, f"{p.name} sets :Speech but no speech_id to match on"


def test_nodes_are_projected_before_the_relationships_that_match_them():
    """Relationship projections MATCH rather than MERGE their endpoints, so list order is
    load-bearing, not cosmetic — every node projection has to run first."""
    last_node = max(IDS.index(p.name) for p in NODE_PROJECTIONS)
    first_rel = min(IDS.index(p.name) for p in REL_PROJECTIONS)
    assert last_node < first_rel


def test_projection_names_are_unique():
    """rebuild() keys its returned counts by name; a duplicate would silently overwrite."""
    assert len(IDS) == len(set(IDS))


@pytest.mark.parametrize("sql,name",
                         [(p.sql, p.name) for p in PROJECTIONS] + [(SPEECH_TEXT_SQL, "SPEECH_TEXT")])
def test_queries_over_pipeline_output_exclude_unfinished_work(sql, name):
    """Half-processed clips and un-analyzed radio calls must never reach the graph: their
    transcripts are absent or partial, and a projection that forgets the filter looks
    perfectly healthy while publishing them. Enforced here rather than by factoring the
    predicate into a shared constant, so the SQL stays literal and greppable."""
    flat = " ".join(sql.split())
    if re.search(r"\b(FROM|JOIN)\s+clips\b", flat):
        assert "status = 'COMPLETE'" in flat, f"{name}: reads clips without the status filter"
    if re.search(r"\b(FROM|JOIN)\s+radio_calls\b", flat):
        assert "analyzed_at IS NOT NULL" in flat, f"{name}: reads radio_calls unfiltered"
