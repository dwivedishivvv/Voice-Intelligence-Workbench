"""OpenF1 hands back timestamps as ISO-8601 *strings*; asyncpg binds a TIMESTAMPTZ from a
datetime object and rejects the string outright.

This is worth a test rather than a code read because of how quietly it failed. `_persist`
logs and continues on error by design — the endpoint's job is to return OpenF1 data, and a
failed cache write does not make that payload wrong — so the live symptom was an F1 page
that worked perfectly against a database where sessions, laps and radio calls were all
silently writing zero rows. Only `f1_drivers`, the one table with no timestamp column,
persisted at all.
"""
from datetime import datetime, timezone

import pytest

from api.app.routers.f1 import _ts


@pytest.mark.parametrize("raw,expected", [
    # the exact shapes observed from OpenF1's sessions / laps / team_radio endpoints
    ("2024-03-02T15:00:00+00:00", datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)),
    ("2024-03-02T14:13:48.440000+00:00",
     datetime(2024, 3, 2, 14, 13, 48, 440000, tzinfo=timezone.utc)),
    ("2024-03-02T15:03:42.341000+00:00",
     datetime(2024, 3, 2, 15, 3, 42, 341000, tzinfo=timezone.utc)),
])
def test_parses_the_shapes_openf1_actually_returns(raw, expected):
    assert _ts(raw) == expected


def test_result_is_a_datetime_not_a_string():
    """The whole point: asyncpg type-checks the binding, so returning a cleaned-up string
    would fail exactly as the raw one did."""
    assert isinstance(_ts("2024-03-02T15:00:00+00:00"), datetime)


def test_z_suffix_is_accepted():
    """Python 3.10's fromisoformat rejects a trailing 'Z'. The repo targets 3.11+, where it
    is accepted, so this would pass for the wrong reason there — it is here to keep the
    normalisation from being dropped as redundant."""
    assert _ts("2024-03-02T15:00:00Z") == datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)


def test_naive_input_is_treated_as_utc():
    """A TIMESTAMPTZ column needs the offset to be explicit; a naive datetime would be
    interpreted against the server's timezone, silently shifting every lap alignment."""
    got = _ts("2024-03-02T15:00:00")
    assert got.tzinfo is not None and got.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("raw", [None, "", "not-a-timestamp", "2024-13-45T99:99:99"])
def test_unusable_values_become_null_rather_than_raising(raw):
    """NULL is the honest reading — the time really is unknown — and it fails safe: the
    DURING_LAP projection requires a non-null timestamp, so such a row is left off the lap
    timeline instead of being placed on it wrongly. Raising here would instead take down
    the whole batch, losing the rows that were fine."""
    assert _ts(raw) is None
