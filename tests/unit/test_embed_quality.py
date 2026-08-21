"""Fingerprint extraction: which audio gets picked, how it is batched, and the check that
catches a diarization label holding two people.

The encoder is faked — these cover the selection and batching around it, which is where
the silent failures live. A real ECAPA pass would only test speechbrain.
"""
import numpy as np
import pytest
import torch

from worker.stages import embed as embed_mod
from worker.stages.embed import (
    select_segments, encode_many, _split_halves, _noise_power, segment_quality,
    SPLIT_HALF_SUSPECT,
)


class Cfg:
    embed_min_s = 1.5
    embed_target_s = 8.0
    clipping_threshold = 0.99
    reliability_fair = 0.45


class FakeEmbedder:
    """Returns a deterministic embedding per row and records how it was called, so the
    tests can assert both the values and that batching actually happened."""

    def __init__(self):
        self.calls = []

    def encode_batch(self, wavs, wav_lens=None):
        self.calls.append({
            "shape": tuple(wavs.shape),
            "wav_lens": None if wav_lens is None else [round(float(v), 4) for v in wav_lens],
        })
        out = torch.zeros(wavs.shape[0], 1, 3)
        for i in range(wavs.shape[0]):
            out[i, 0, 0] = float(wavs[i].abs().mean())
            out[i, 0, 1] = float(wavs[i].std())
            out[i, 0, 2] = 1.0
        return out


class FakePool:
    def __init__(self):
        self.embedder = FakeEmbedder()


SR = 16000


def tone(seconds, amp=0.2, freq=220.0, seed=0):
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    return (amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(n)).astype(np.float32)


def turn(start, end, label="S0", **kw):
    return {"start": start, "end": end, "label": label,
            "is_overlap": kw.get("is_overlap", False), "too_short": kw.get("too_short", False)}


# ---- batching ----

def test_every_segment_goes_through_the_encoder_in_one_call():
    pool = FakePool()
    segs = [tone(1.0), tone(2.0), tone(0.5)]
    out = encode_many(pool, segs)
    assert len(pool.embedder.calls) == 1, "one padded batch, not one call per segment"
    assert out.shape == (3, 3)


def test_padding_is_declared_so_the_encoder_ignores_it():
    """Without wav_lens the zero-padded tail is pooled in as if the speaker had produced
    that silence, which shortens and flattens every embedding but the longest."""
    pool = FakePool()
    out = encode_many(pool, [tone(2.0), tone(1.0)])
    call = pool.embedder.calls[0]
    assert call["shape"] == (2, 2 * SR)
    assert call["wav_lens"] == [1.0, 0.5]
    assert out.shape == (2, 3)


def test_embeddings_come_back_unit_length():
    out = encode_many(FakePool(), [tone(1.0), tone(2.0)])
    for row in out:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)


# ---- selection ----

def test_overlapped_and_too_short_turns_are_never_fingerprinted():
    """Overlap means two voices in the audio; too_short means diarization kept the turn
    only so its words could be attributed (stages/diarize.py)."""
    audio = tone(20.0)
    turns = [turn(0.0, 5.0), turn(6.0, 11.0, is_overlap=True), turn(12.0, 17.0, too_short=True)]
    chunks, total = select_segments(audio, turns, Cfg, noise_power=1e-6)
    assert len(chunks) == 1 and total == pytest.approx(5.0, abs=0.01)


def test_a_clean_turn_outranks_a_clipped_one_of_similar_length():
    """The old sort was on duration alone, so a long clipped turn beat a comparable clean
    one for no reason other than length."""
    audio = np.concatenate([tone(4.0, amp=0.2, seed=1), np.ones(int(4.2 * SR), dtype=np.float32)])
    turns = [turn(0.0, 4.0), turn(4.0, 8.2)]          # clean 4.0s, saturated 4.2s
    chunks, _total = select_segments(audio, turns, Cfg, noise_power=1e-8)
    assert float(np.abs(chunks[0]).max()) < 1.0, "the saturated turn was ranked first"


def test_duration_still_dominates_between_similar_quality_turns():
    audio = np.concatenate([tone(2.0, seed=2), tone(6.0, seed=3)])
    turns = [turn(0.0, 2.0), turn(2.0, 8.0)]
    chunks, _ = select_segments(audio, turns, Cfg, noise_power=1e-8)
    assert chunks[0].size > 4 * SR


def test_a_label_below_the_floor_yields_nothing_rather_than_a_weak_embedding():
    audio = tone(5.0)
    chunks, total = select_segments(audio, [turn(0.0, 0.8)], Cfg, noise_power=1e-8)
    assert chunks == [] and total == pytest.approx(0.8, abs=0.01)


def test_segment_quality_penalizes_noise_and_clipping():
    clean = tone(2.0, amp=0.2, seed=4)
    saturated = np.ones(int(2.0 * SR), dtype=np.float32)
    assert segment_quality(clean, 1e-8, Cfg) > segment_quality(saturated, 1e-8, Cfg)
    # same audio, louder noise floor -> lower quality
    assert segment_quality(clean, 1e-8, Cfg) > segment_quality(clean, 1e-2, Cfg)


# ---- split-half contamination check ----

def test_halves_are_split_across_turns_not_mid_concatenation():
    """Cutting the concatenation in the middle would put both halves inside the same few
    turns, so the two embeddings would agree for reasons unrelated to identity."""
    chunks = [tone(2.0, seed=5), tone(2.0, seed=6), tone(2.0, seed=7)]
    a, b = _split_halves(chunks)
    assert a.size + b.size == sum(c.size for c in chunks)
    assert a.size > 0 and b.size > 0


def test_too_little_audio_to_split_returns_nothing():
    assert _split_halves([tone(1.0)]) is None                  # one turn, nowhere to cut
    assert _split_halves([tone(0.5), tone(0.5)]) is None       # under SPLIT_HALF_MIN_S


class FakeCtx:
    """Just enough of worker.ctx.Ctx for the embed stage."""

    def __init__(self, audio, turns):
        self.audio_norm, self.turns, self.vad = audio, turns, []
        self.cfg, self.pool = Cfg, FakePool()
        self.speakers, self.warnings = {}, []

    def warn(self, code, **kw):
        self.warnings.append({"code": code, **kw})


def _run_embed_with(monkeypatch, half_a, half_b):
    """Drive run() with controlled embeddings so the halves' agreement is the variable.

    Whether real ECAPA separates two voices is speechbrain's property, not this
    codebase's; what belongs under test here is what the stage *does* with the answer.
    """
    import asyncio

    full = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rows = [full, np.asarray(half_a, dtype=np.float32), np.asarray(half_b, dtype=np.float32)]
    monkeypatch.setattr(embed_mod, "encode_many", lambda pool, segs: np.stack(rows[:len(segs)]))

    audio = tone(12.0)
    ctx = FakeCtx(audio, [turn(0.0, 5.0), turn(5.0, 10.0)])
    asyncio.run(embed_mod.run(ctx))
    return ctx


def test_disagreeing_halves_discount_reliability_and_raise_a_warning(monkeypatch):
    """Two voices blended into one embedding is the case that produces a confident *wrong*
    name — it can sit plausibly near a third profile. Nothing detected it before."""
    ctx = _run_embed_with(monkeypatch, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])   # orthogonal
    sp = ctx.speakers["S0"]
    assert sp.split_half_sim == pytest.approx(0.0, abs=1e-6)
    assert sp.reliability == 0.0
    assert sp.reliability_reason == "split_half_disagreement"
    assert any(w["code"] == "MIXED_LABEL_SUSPECTED" for w in ctx.warnings)


def test_agreeing_halves_leave_reliability_untouched(monkeypatch):
    ctx = _run_embed_with(monkeypatch, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])   # identical
    sp = ctx.speakers["S0"]
    assert sp.split_half_sim == pytest.approx(1.0, abs=1e-6)
    assert sp.reliability > 0.0
    assert sp.reliability_reason is None
    assert not any(w["code"] == "MIXED_LABEL_SUSPECTED" for w in ctx.warnings)


# ---- noise floor ----

def test_noise_power_is_measured_outside_the_speech_regions():
    speech = tone(2.0, amp=0.5, seed=11)
    silence = (0.001 * np.random.default_rng(12).standard_normal(2 * SR)).astype(np.float32)
    audio = np.concatenate([speech, silence])
    p = _noise_power(audio, [(0.0, 2.0)])
    assert p < float(np.mean(speech ** 2))


def test_a_clip_with_no_silence_falls_back_instead_of_reporting_absurd_snr():
    audio = tone(2.0)
    assert _noise_power(audio, [(0.0, 2.0)]) == 1e-12
