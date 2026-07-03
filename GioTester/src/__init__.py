"""GioTester public API.

Strategy authors only need:
    from src import OrderCommand, OrderType

Backtest drivers use:
    from src import BacktestConfig, DataConfig, DataLoader, run_backtest,
                    log_results, print_result_summary
"""

from .config import BacktestConfig, DataConfig
from .data_loader import DataLoader, MarketData
from .position import (
    CloseReason,
    ExecutionEvent,
    FundingEvent,
    LiquidationEvent,
    OrderCommand,
    OrderRejectedEvent,
    OrderType,
    Position,
)
from .result import SimResult, log_results, print_result_summary
from .runner import ENGINE_SEMANTICS_VERSION, run_backtest, run_backtest_from_trader_file
from .state import MarketState, StateBucket
from .strategy_loader import load_trader

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "DataLoader",
    "MarketData",
    "MarketState",
    "StateBucket",
    "Position",
    "OrderCommand",
    "OrderType",
    "CloseReason",
    "ExecutionEvent",
    "FundingEvent",
    "OrderRejectedEvent",
    "LiquidationEvent",
    "SimResult",
    "ENGINE_SEMANTICS_VERSION",
    "run_backtest",
    "run_backtest_from_trader_file",
    "load_trader",
    "log_results",
    "print_result_summary",
]
