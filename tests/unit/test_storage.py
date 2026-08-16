"""Deleting a clip removes a mix of files and directories, and the two need different
calls. Getting that wrong is not a quiet failure — it raised out of the delete endpoint as
a 500 and left every artifact on disk — but it is invisible until a clip that has actually
been processed is deleted, because an unprocessed clip has no work directory to trip over.
"""
from types import SimpleNamespace

import pytest

from common.storage import delete_clip_files

CLIP = "c6683340-8b6b-4701-9132-c083ff297a8a"
OTHER = "aaaaaaaa-0000-0000-0000-000000000000"


@pytest.fixture
def cfg(tmp_path):
    """A DATA_DIR laid out the way the pipeline leaves it after a successful run."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "work" / CLIP).mkdir(parents=True)
    (tmp_path / "exports" / CLIP).mkdir(parents=True)

    (tmp_path / "raw" / f"{CLIP}.wav").write_bytes(b"audio")
    (tmp_path / "work" / CLIP / "processed.wav").write_bytes(b"audio")
    (tmp_path / "work" / CLIP / "vad.json").write_text("{}")
    (tmp_path / "exports" / CLIP / "transcript.srt").write_text("1\n")
    # a second clip that must survive
    (tmp_path / "raw" / f"{OTHER}.wav").write_bytes(b"audio")
    (tmp_path / "work" / OTHER).mkdir()
    return SimpleNamespace(data_dir=str(tmp_path))


def test_deletes_files_and_directories_together(cfg, tmp_path):
    """The work directory is what breaks a naive implementation: the glob matches it, and
    unlink() on a directory raises rather than removing it."""
    delete_clip_files(cfg, CLIP)

    assert not (tmp_path / "raw" / f"{CLIP}.wav").exists()
    assert not (tmp_path / "work" / CLIP).exists(), "work directory survived the delete"
    assert not (tmp_path / "exports" / CLIP).exists(), "exports directory survived"


def test_leaves_other_clips_alone(cfg, tmp_path):
    """The glob is prefix-based, so a delete must not reach past its own clip id."""
    delete_clip_files(cfg, CLIP)

    assert (tmp_path / "raw" / f"{OTHER}.wav").exists()
    assert (tmp_path / "work" / OTHER).exists()


def test_deleting_twice_is_not_an_error(cfg):
    """The endpoint writes a deletion receipt after the files are gone, so a retry after a
    partial failure must not raise on the second pass."""
    delete_clip_files(cfg, CLIP)
    delete_clip_files(cfg, CLIP)


def test_deleting_a_clip_with_no_files_is_not_an_error(cfg):
    """A clip rejected at validation never gets a work directory."""
    delete_clip_files(cfg, "00000000-0000-0000-0000-000000000000")
