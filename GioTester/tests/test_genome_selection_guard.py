"""Phase 0h: one-sided genome guards + overlap-drop coverage for select_longs_shorts."""
from __future__ import annotations

from src.genome.adapter import select_longs_shorts

SCORES = {"AAA": 0.90, "BBB": 0.80, "EEE": 0.40, "CCC": 0.20, "DDD": 0.05}


def test_short_only_n_long_zero_selects_no_longs():
    longs, shorts = select_longs_shorts(SCORES, 0, 2, held=set())
    assert longs == []
    assert set(shorts) == {"CCC", "DDD"}


def test_long_only_n_short_zero_selects_no_shorts():
    longs, shorts = select_longs_shorts(SCORES, 2, 0, held=set())
    assert shorts == []
    assert set(longs) == {"AAA", "BBB"}


def test_single_candidate_one_sided_is_allowed():
    longs, shorts = select_longs_shorts({"AAA": 0.9}, 1, 0, held=set())
    assert longs == ["AAA"] and shorts == []
    longs, shorts = select_longs_shorts({"AAA": 0.9}, 0, 1, held=set())
    assert longs == [] and shorts == ["AAA"]


def test_single_candidate_two_sided_still_blocked():
    # preserves legacy SLTP behavior: two-sided books need >= 2 candidates
    assert select_longs_shorts({"AAA": 0.9}, 2, 2, held=set()) == ([], [])


def test_empty_scores():
    assert select_longs_shorts({}, 0, 2, held=set()) == ([], [])


def test_overlap_drop_long_wins():
    # 3 candidates, n=2/2: top-2 longs, bottom-2 shorts intersect on middle asset
    scores = {"AAA": 0.9, "BBB": 0.5, "CCC": 0.1}
    longs, shorts = select_longs_shorts(scores, 2, 2, held=set())
    assert set(longs) == {"AAA", "BBB"}
    assert shorts == ["CCC"]  # BBB dropped from shorts, long wins
