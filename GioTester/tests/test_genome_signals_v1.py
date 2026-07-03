# tests/test_genome_signals_v1.py
"""Task T2: signal + timing genes (rank_30d, cached rank, momentum,
funding_carry, delay) and the ComposedTrader observe() hook."""
from __future__ import annotations

import src.genome.library  # noqa: F401  (registers genes on import)
from src.genome.adapter import GeneSpec, Genome, build_trader
from src.genome.registry import build_gene
from strategy_harness import make_state, market_view


def _release_state(bar_index=0):
    ranks = {
        "AAA": (0.90, 0.10),
        "BBB": (0.80, 0.20),
        "CCC": (0.20, 0.70),
        "DDD": (0.05, 0.95),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def _non_release_state(bar_index):
    market = {a: market_view(a, 100.0) for a in ["AAA", "BBB", "CCC", "DDD"]}
    return make_state(bar_index=bar_index, market=market, ranks=None, is_release=False)


# --- rank_30d ---------------------------------------------------------------


def test_rank_30d_uses_field_index_1():
    gene = build_gene("signal", "rank_30d", {})
    state = _release_state()
    scores = gene.score(state, set(state.market))
    assert scores["AAA"] == 0.10
    assert scores["DDD"] == 0.95


# --- cached rank signals -----------------------------------------------------


def test_rank_cached_survives_off_release_bar():
    gene = build_gene("signal", "rank_cached", {})
    release = _release_state(bar_index=0)
    gene.observe(release)
    later = _non_release_state(bar_index=5)
    scores = gene.score(later, set(later.market))
    assert scores["AAA"] == 0.90


def test_rank_30d_cached_survives_off_release_bar():
    gene = build_gene("signal", "rank_30d_cached", {})
    release = _release_state(bar_index=0)
    gene.observe(release)
    later = _non_release_state(bar_index=5)
    scores = gene.score(later, set(later.market))
    assert scores["AAA"] == 0.10
    assert scores["DDD"] == 0.95


def test_plain_rank_returns_empty_off_release_contrast():
    gene = build_gene("signal", "rank", {})
    later = _non_release_state(bar_index=5)
    assert gene.score(later, set(later.market)) == {}


def test_cached_rank_before_any_observe_is_empty():
    gene = build_gene("signal", "rank_cached", {})
    later = _non_release_state(bar_index=5)
    assert gene.score(later, set(later.market)) == {}


# --- momentum -----------------------------------------------------------------


def _state_with_prices(bar_index, prices):
    market = {a: market_view(a, px) for a, px in prices.items()}
    return make_state(bar_index=bar_index, market=market, ranks=None, is_release=False)


def test_momentum_full_window_math():
    gene = build_gene("signal", "momentum", {"lookback_bars": 3})
    # bars 0..3 -> window of 4 (lookback_bars + 1) is full at bar 3
    prices_seq = [100.0, 101.0, 102.0, 110.0]
    for i, px in enumerate(prices_seq):
        gene.observe(_state_with_prices(i, {"AAA": px}))
    state = _state_with_prices(3, {"AAA": 110.0})
    scores = gene.score(state, {"AAA"})
    assert abs(scores["AAA"] - (110.0 / 100.0 - 1.0)) < 1e-12


def test_momentum_insufficient_window_skipped():
    gene = build_gene("signal", "momentum", {"lookback_bars": 5})
    for i, px in enumerate([100.0, 101.0, 102.0]):
        gene.observe(_state_with_prices(i, {"AAA": px}))
    state = _state_with_prices(2, {"AAA": 102.0})
    scores = gene.score(state, {"AAA"})
    assert "AAA" not in scores


def test_momentum_filters_by_universe():
    gene = build_gene("signal", "momentum", {"lookback_bars": 1})
    gene.observe(_state_with_prices(0, {"AAA": 100.0, "BBB": 50.0}))
    gene.observe(_state_with_prices(1, {"AAA": 110.0, "BBB": 55.0}))
    state = _state_with_prices(1, {"AAA": 110.0, "BBB": 55.0})
    scores = gene.score(state, {"AAA"})
    assert set(scores) == {"AAA"}
    assert abs(scores["AAA"] - 0.10) < 1e-12


# --- funding_carry --------------------------------------------------------


def test_funding_carry_sign_is_negative_rate():
    gene = build_gene("signal", "funding_carry", {})
    market = {
        "AAA": market_view("AAA", 100.0, funding_rate=0.001),
        "BBB": market_view("BBB", 100.0, funding_rate=-0.002),
    }
    state = make_state(market=market)
    scores = gene.score(state, set(market))
    assert scores["AAA"] == -0.001
    assert scores["BBB"] == 0.002


def test_funding_carry_skips_non_finite():
    gene = build_gene("signal", "funding_carry", {})
    market = {
        "AAA": market_view("AAA", 100.0, funding_rate=float("nan")),
        "BBB": market_view("BBB", 100.0, funding_rate=0.001),
    }
    state = make_state(market=market)
    scores = gene.score(state, set(market))
    assert set(scores) == {"BBB"}


# --- delay timing -----------------------------------------------------------


def test_delay_fires_exactly_at_k_bars_post_release():
    gene = build_gene("entry_timing", "delay", {"bars_after_release": 24})
    release = _release_state(bar_index=0)
    gene.observe(release)

    before = _non_release_state(bar_index=23)
    at_k = _non_release_state(bar_index=24)
    after = _non_release_state(bar_index=25)

    assert gene.should_enter(before) is False
    assert gene.should_enter(at_k) is True
    assert gene.should_enter(after) is False


def test_delay_rearms_on_next_release():
    gene = build_gene("entry_timing", "delay", {"bars_after_release": 24})
    gene.observe(_release_state(bar_index=0))
    assert gene.should_enter(_non_release_state(bar_index=24)) is True

    # a fresh release bar re-anchors the count
    gene.observe(_release_state(bar_index=30))
    assert gene.should_enter(_non_release_state(bar_index=54)) is True
    assert gene.should_enter(_non_release_state(bar_index=24)) is False


def test_delay_never_fires_without_a_release_observed():
    gene = build_gene("entry_timing", "delay", {"bars_after_release": 24})
    assert gene.should_enter(_non_release_state(bar_index=24)) is False


# --- adapter observe hook ----------------------------------------------------


class _SpyGene:
    """Minimal stub implementing every protocol method as a no-op, plus a
    counting observe() -- used to verify the ComposedTrader observe hook
    calls every gene that defines observe() on every bar."""

    def __init__(self):
        self.observed_bars = []

    def observe(self, state):
        self.observed_bars.append(state.bar_index)

    # universe_filter
    def eligible(self, state):
        return set(state.market)

    # signal
    def score(self, state, universe):
        return {}

    # entry_timing
    def should_enter(self, state):
        return False

    # sizing
    def orders_for(self, state, longs, shorts):
        return []

    # exit_rule
    def exits(self, state, ledger):
        return []


def _spy_genome():
    return Genome(
        name="SpyGenome",
        universe_filter=GeneSpec("all_tradable"),
        signal=GeneSpec("rank"),
        entry_timing=GeneSpec("release_bar"),
        sizing=GeneSpec("fixed_notional"),
        exit_rule=GeneSpec("bracket"),
    )


def test_observe_hook_called_every_bar_for_every_gene_defining_it():
    genome = _spy_genome()
    trader = build_trader(genome)

    spies = [_SpyGene(), _SpyGene(), _SpyGene(), _SpyGene(), _SpyGene()]
    trader._uf, trader._sig, trader._timing, trader._sizing, trader._exit = spies

    bars = [_non_release_state(0), _release_state(1), _non_release_state(2)]
    for state in bars:
        trader.run(state)

    expected = [0, 1, 2]
    for spy in spies:
        assert spy.observed_bars == expected
