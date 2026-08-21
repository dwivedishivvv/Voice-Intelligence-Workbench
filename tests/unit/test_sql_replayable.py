"""db/init.sql only ever runs on a *fresh* postgres volume, so races.sql and f1.sql are
written to be replayable against a live database instead. Nothing enforces that at
runtime: a non-idempotent statement slipped into either file works fine on a fresh
volume and only fails much later, on someone's populated database, during the replay
that was supposed to be the safe path. These assert the property directly."""
import re
from pathlib import Path

import pytest

DB_DIR = Path(__file__).resolve().parents[2] / "db"
REPLAYABLE = ["races.sql", "f1.sql", "ontrack.sql", "speaker_quality.sql"]


def _statements(name: str) -> list[str]:
    """Whitespace-normalized, uppercased statements with `--` comments stripped, so prose
    about CREATE TABLE (and any commented-out DDL) isn't mistaken for the real thing."""
    text = re.sub(r"--[^\n]*", "", (DB_DIR / name).read_text())
    return [" ".join(s.split()).upper() for s in text.split(";") if s.strip()]


@pytest.mark.parametrize("name", REPLAYABLE)
def test_replayable_ddl_is_idempotent(name):
    for stmt in _statements(name):
        if stmt.startswith("CREATE TABLE") or stmt.startswith("CREATE INDEX") \
                or stmt.startswith("CREATE UNIQUE INDEX"):
            assert "IF NOT EXISTS" in stmt, f"{name}: not replayable -> {stmt[:70]}"
        if stmt.startswith("ALTER TABLE") and "ADD COLUMN" in stmt:
            assert "ADD COLUMN IF NOT EXISTS" in stmt, f"{name}: not replayable -> {stmt[:70]}"
        if stmt.startswith("ALTER TYPE") and "ADD VALUE" in stmt:
            assert "ADD VALUE IF NOT EXISTS" in stmt, f"{name}: not replayable -> {stmt[:70]}"


@pytest.mark.parametrize("name", REPLAYABLE)
def test_init_includes_every_replayable_file(name):
    """The other half of the contract: being replayable is no use if a fresh volume never
    gets the file at all, which is silent until something queries a missing table.

    Matches any psql include directive rather than one exact spelling. The include started
    as a relative `\ir` and had to become an absolute `\i /db/...` -- `\ir` resolves
    against the including script's own directory, and inside the container init.sql sits in
    initdb.d while its siblings are mounted elsewhere. Asserting the invariant (this file
    gets included) rather than the mechanism keeps the test from failing on a correct fix."""
    init = (DB_DIR / "init.sql").read_text()
    includes = re.findall(r"^\s*\\ir?\s+(\S+)", init, re.MULTILINE)
    assert any(inc.rsplit("/", 1)[-1] == name for inc in includes), (
        f"{name} is never included by init.sql; found includes: {includes}")
