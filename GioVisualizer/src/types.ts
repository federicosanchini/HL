export interface Metrics {
  PnL: number;
  DD: number;
  Sharpe: number;
  Sortino: number;
}

export interface LiquidationEvent {
  timestamp: string;
  asset: string;
}

export interface BacktestResult {
  strategy: string;
  timeline: string[];
  total_equity: number[];
  per_perp_equity: Record<string, number[]>;
  per_perp_position: Record<string, number[]>;
  metrics_total: Metrics;
  metrics_per_perp: Record<string, Metrics>;
  n_opened: number;
  n_closed: number;
  n_liquidated: number;
  liquidation_events: LiquidationEvent[];
}
