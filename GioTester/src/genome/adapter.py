# src/genome/adapter.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from src import OrderCommand

from .genes import EntryLedger
from .registry import build_gene


@dataclass(frozen=True)
class GeneSpec:
    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Genome:
    name: str
    universe_filter: GeneSpec
    signal: GeneSpec
    entry_timing: GeneSpec
    sizing: GeneSpec
    exit_rule: GeneSpec
    n_long: int = 2
    n_short: int = 2
    margin_mode: str = "cross"
    leverage: float = 1.0
    min_notional_usd: float = 10.0


def select_longs_shorts(
    scores: Dict[str, float],
    n_long: int,
    n_short: int,
    held: Set[str],
) -> Tuple[List[str], List[str]]:
    cands = sorted(scores.items(), key=lambda t: (t[1], t[0]))
    # two-sided books need >= 2 candidates (legacy SLTP semantics);
    # one-sided books are legitimate with a single candidate
    two_sided = n_long > 0 and n_short > 0
    if not cands or (two_sided and len(cands) < 2):
        return [], []
    # n <= 0 must yield an empty side: cands[-0:] is the WHOLE list
    shorts = [a for a, _ in cands[:n_short]] if n_short > 0 else []
    longs = [a for a, _ in cands[-n_long:]] if n_long > 0 else []
    long_set = set(longs)
    shorts = [a for a in shorts if a not in long_set]
    longs = [a for a in longs if a not in held]
    shorts = [a for a in shorts if a not in held]
    return longs, shorts


class ComposedTrader:
    """A Trader assembled from five genes; run(state) drives a fixed pipeline."""

    def __init__(self, genome: Genome, uf, sig, timing, sizing, exit_rule) -> None:
        self.name = genome.name
        self.margin_mode = genome.margin_mode
        self._genome = genome
        self._uf = uf
        self._sig = sig
        self._timing = timing
        self._sizing = sizing
        self._exit = exit_rule
        self._ledger = EntryLedger()

    def run(self, state) -> List[OrderCommand]:
        # Observe hook (Task T2): stateful genes (momentum signal, cached-rank
        # signals, delay timing, ...) need every bar, not just the bars their
        # score()/should_enter() are consulted on. Fixed order, additive,
        # optional (hasattr) -- does not change the existing pipeline contract.
        for gene in (self._uf, self._sig, self._timing, self._sizing, self._exit):
            if hasattr(gene, "observe"):
                gene.observe(state)
        self._ledger.prune(set(state.positions))
        # R7 level anchoring: refresh each held position's ledger price from the
        # realized entry_price (the actual open[i+1] fill) every bar. This is
        # idempotent — entry_price only moves on adds/flips, which would reset
        # exit-gene tracking anyway — so an unconditional per-bar refresh is
        # simpler than tracking a "first observation" flag and behaves identically.
        for asset, pos in state.positions.items():
            rec = self._ledger.get(asset)
            if rec is not None:
                # `record()` overwrites in place; passing back the existing
                # bar_index preserves the expiry anchor while refreshing price.
                self._ledger.record(asset, rec.bar_index, pos.entry_price)
        orders: List[OrderCommand] = list(self._exit.exits(state, self._ledger))
        if self._timing.should_enter(state):
            universe = self._uf.eligible(state)
            scores = self._sig.score(state, universe)
            held = set(state.positions)
            longs, shorts = select_longs_shorts(
                scores, self._genome.n_long, self._genome.n_short, held
            )
            entry_orders = self._sizing.orders_for(state, longs, shorts)
            for order in entry_orders:
                mv = state.market.get(order.asset)
                if mv is not None:
                    self._ledger.record(order.asset, state.bar_index, mv.mark_px)
            orders.extend(entry_orders)
        return orders


def _merged_params(genome: Genome, spec: GeneSpec) -> dict:
    """Merge genome-level knobs into a slot's params (genome overrides on collision)."""
    return {
        **spec.params,
        "leverage": genome.leverage,
        "min_notional_usd": genome.min_notional_usd,
    }


def build_trader(genome: Genome) -> ComposedTrader:
    uf = build_gene(
        "universe_filter", genome.universe_filter.kind, _merged_params(genome, genome.universe_filter)
    )
    sig = build_gene("signal", genome.signal.kind, _merged_params(genome, genome.signal))
    timing = build_gene(
        "entry_timing", genome.entry_timing.kind, _merged_params(genome, genome.entry_timing)
    )
    sizing = build_gene("sizing", genome.sizing.kind, _merged_params(genome, genome.sizing))
    exit_rule = build_gene("exit_rule", genome.exit_rule.kind, _merged_params(genome, genome.exit_rule))
    return ComposedTrader(genome, uf, sig, timing, sizing, exit_rule)
