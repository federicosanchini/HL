# GioVisualizer

Static React app for visualizing HyperLiquid backtest results. Drop a JSON file exported by `GioTester` and get interactive charts.

---

## Prerequisites

- **Node.js 18+** — download from https://nodejs.org
- **Python backtester** — `GioTester/` must be functional (already set up)

---

## 1 — Generate a backtest JSON

```bash
cd GioTester
python tester.py
```

Output: `GioTester/results/<strategy_name>.json` (one file per strategy).

---

## 2 — Run locally

```bash
cd GioVisualizer
npm install       # first time only — installs ~200 MB of deps
npm run dev       # starts dev server at http://localhost:5173
```

Open the URL, drop a JSON file from `GioTester/results/`, and the charts render immediately.

---

## 3 — Deploy to Netlify (online, shareable URL)

### First-time setup

1. Create a free account at https://netlify.com
2. "Add new site" → "Import an existing project" → GitHub
3. Select repo `federicosanchini/HL`
4. Netlify reads `netlify.toml` automatically — no manual config needed
5. Click **Deploy**

### Auto-deploy

Every push to branch `giorgio` triggers a new build and updates the live URL.

### Manual deploy (no CI)

```bash
cd GioVisualizer
npm run build            # produces dist/
npx netlify-cli deploy --prod --dir dist
```

Or drag `GioVisualizer/dist/` to https://app.netlify.com/drop .

---

## JSON format (produced by `log_results()`)

```jsonc
{
  "strategy": "HODL_10",
  "timeline": ["2024-01-01T00:00:00+00:00", ...],  // ISO-8601 UTC
  "total_equity": [2000.0, 2001.3, ...],             // one value per bar
  "per_perp_equity": {
    "BTC": [0.0, 0.5, ...],                          // PnL contribution per bar
    "ETH": [0.0, -0.2, ...]
  },
  "metrics_total": { "PnL": 150.0, "DD": -0.05, "Sharpe": 1.2, "Sortino": 1.8 },
  "metrics_per_perp": {
    "BTC": { "PnL": 80.0, "DD": -0.03, "Sharpe": 1.1, "Sortino": 1.5 }
  },
  "n_opened": 40,
  "n_closed": 38,
  "n_liquidated": 0
}
```

Only perps that actually traded appear in `per_perp_equity` and `metrics_per_perp`.

---

## Chart layout

| Row | Left | Right |
|-----|------|-------|
| 1 | Cumulative portfolio equity | All per-asset PnL contributions overlapped |
| 2+ | Asset 1 PnL | Asset 2 PnL |
| … | … | … |

Charts are responsive — resize the browser window freely.

---

## File structure

```txt
GioVisualizer/
├── src/
│   ├── components/
│   │   ├── Charts.tsx        # Plotly charts grid
│   │   ├── DropZone.tsx      # Mantine file drop
│   │   └── MetricsTable.tsx  # PnL/DD/Sharpe/Sortino table
│   ├── App.tsx
│   ├── main.tsx
│   └── types.ts              # BacktestResult interface
├── index.html
├── netlify.toml              # Netlify deploy config
├── package.json
└── vite.config.ts
```
