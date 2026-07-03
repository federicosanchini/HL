from __future__ import annotations

import math

from strategy_harness import load, make_state, market_view, position_view


def _release_state(bar_index=0):
    # Five ranked assets; ascending signal order DDD < CCC < EEE < BBB < AAA.
    ranks = {
        "AAA": (0.90, 0.0),
        "BBB": (0.80, 0.0),
        "EEE": (0.40, 0.0),
        "CCC": (0.20, 0.0),
        "DDD": (0.05, 0.0),
    }
    market = {a: market_view(a, 100.0) for a in ranks}
    return make_state(bar_index=bar_index, market=market, ranks=ranks, is_release=True)


def test_entry_longs_top_shorts_bottom_n2():
    t = load("SLTP_Bracket.py")
    orders = t.run(_release_state())
    longs = {o.asset for o in orders if o.side == 1}
    shorts = {o.asset for o in orders if o.side == -1}
    assert longs == {"AAA", "BBB"}
    assert shorts == {"CCC", "DDD"}
    assert all(o.order_type == "market" and not o.reduce_only for o in orders)
    assert all(o.notional == 10.0 for o in orders)


def test_entry_skips_already_held():
    t = load("SLTP_Bracket.py")
    # AAA already held -> not re-entered; BBB still entered long.
    state = _release_state()
    state = make_state(
        bar_index=0,
        market=state.market,
        ranks=state.current_ranks_row,
        is_release=True,
        positions={"AAA": position_view("AAA", 0.1, 100.0, 100.0)},
    )
    orders = t.run(state)
    longs = {o.asset for o in orders if o.side == 1}
    assert "AAA" not in longs
    assert "BBB" in longs


def _prime_long(trader, asset="AAA", entry=100.0):
    """Run an entry release bar so the trader records entry_px for `asset`."""
    trader.run(_release_state(bar_index=0))
    assert math.isclose(trader._entry_px[asset], entry)


def _exit_state(asset, size, entry, mark, bar_index=5):
    return make_state(
        bar_index=bar_index,
        market={asset: market_view(asset, mark)},
        positions={asset: position_view(asset, size, entry, mark)},
        is_release=False,
    )


def test_bracket_long_take_profit():
    t = load("SLTP_Bracket.py")  # sl=0.05, tp=0.10
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 111.0))  # +11%
    assert len(orders) == 1 and orders[0].reduce_only and orders[0].side == -1


def test_bracket_long_stop_loss():
    t = load("SLTP_Bracket.py")
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0))  # -6%, mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_bracket_long_inside_band_no_exit():
    t = load("SLTP_Bracket.py")
    _prime_long(t)
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 103.0))  # +3%
    assert orders == []


def test_bracket_short_take_profit():
    t = load("SLTP_Bracket.py")
    t.run(_release_state(bar_index=0))  # primes DDD short at 100
    assert math.isclose(t._entry_px["DDD"], 100.0)
    orders = t.run(_exit_state("DDD", -0.2, 100.0, 88.0))  # short +12% favorable, notional 17.6 >= min
    assert len(orders) == 1 and orders[0].reduce_only and orders[0].side == 1


def test_expiry_forces_close():
    t = load("SLTP_Bracket.py", bars_per_day=24)  # expiry_bars = 240
    _prime_long(t)
    # inside band but past expiry -> close
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 102.0, bar_index=240))
    assert len(orders) == 1 and orders[0].reduce_only


def test_exit_skipped_below_min_notional():
    t = load("SLTP_Bracket.py", min_notional_usd=10.0)
    _prime_long(t)
    # size*mark = 0.05*94 = 4.7 < 10 -> no order despite stop breach
    orders = t.run(_exit_state("AAA", 0.05, 100.0, 94.0))
    assert orders == []


def test_stop_only_closes_on_stop():
    t = load("SL_Only.py")  # sl=0.05, no TP
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0))  # -6%, mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_stop_only_lets_winner_run():
    t = load("SL_Only.py")
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 130.0))  # +30%, no TP
    assert orders == []


def test_target_only_closes_on_target():
    t = load("TP_Only.py")  # tp=0.10, no SL
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 111.0))  # +11%
    assert len(orders) == 1 and orders[0].reduce_only


def test_target_only_rides_loser():
    t = load("TP_Only.py")
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 70.0))  # -30%, no SL
    assert orders == []


def test_asym_tight_stop_fires():
    t = load("SLTP_Asym.py")  # sl=0.04, tp=0.12
    t.run(_release_state(bar_index=0))
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 95.5))  # -4.5%, mark notional 19.1 >= min
    assert len(orders) == 1 and orders[0].reduce_only


def test_asym_wide_target_holds_then_fires():
    t = load("SLTP_Asym.py")
    t.run(_release_state(bar_index=0))
    assert t.run(_exit_state("AAA", 0.1, 100.0, 108.0)) == []   # +8% < 12%
    assert len(t.run(_exit_state("AAA", 0.1, 100.0, 113.0))) == 1  # +13%


def test_trailing_ratchets_and_gives_back():
    t = load("Trailing.py")  # trail=0.05
    t.run(_release_state(bar_index=0))  # entry AAA @ 100
    # climb to +10% (peak), no exit
    assert t.run(_exit_state("AAA", 0.1, 100.0, 110.0, bar_index=1)) == []
    # give back to +6%: 10% - 6% = 4% < 5% -> no exit
    assert t.run(_exit_state("AAA", 0.1, 100.0, 106.0, bar_index=2)) == []
    # give back to +4%: 10% - 4% = 6% >= 5% -> exit
    orders = t.run(_exit_state("AAA", 0.1, 100.0, 104.0, bar_index=3))
    assert len(orders) == 1 and orders[0].reduce_only


def test_trailing_initial_stop():
    t = load("Trailing.py")
    t.run(_release_state(bar_index=0))  # entry AAA @ 100
    # never profitable; peak stays 0; -6% <= 0 - 5% -> exit
    orders = t.run(_exit_state("AAA", 0.2, 100.0, 94.0, bar_index=1))  # mark notional 18.8 >= min
    assert len(orders) == 1 and orders[0].reduce_only
