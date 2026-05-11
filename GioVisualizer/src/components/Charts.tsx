import type { Layout } from 'plotly.js';
import Plot from 'react-plotly.js';
import type { BacktestResult } from '../types';

const COLORS = [
  '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
];

const makeLayout = (title: string, yLabel: string, showLegend = false): Partial<Layout> => ({
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#C1C2C5' },
  margin: { t: 40, r: 20, b: 60, l: 70 },
  title: { text: title, font: { color: '#C1C2C5', size: 14 } },
  xaxis: { gridcolor: '#2C2E33', color: '#C1C2C5', zeroline: false },
  yaxis: { gridcolor: '#2C2E33', color: '#C1C2C5', zeroline: false, title: { text: yLabel } },
  showlegend: showLegend,
  legend: { font: { color: '#C1C2C5' } },
});

const GRID: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '16px',
  marginBottom: '16px',
};

const PLOT_STYLE: React.CSSProperties = { width: '100%' };
const CONFIG = { responsive: true, displaylogo: false };

interface Props {
  result: BacktestResult;
}

export default function Charts({ result }: Props) {
  const perps = Object.keys(result.per_perp_equity).sort();
  const t = result.timeline;

  const perpTrace = (p: string, idx: number) => ({
    x: t,
    y: result.per_perp_equity[p],
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: p,
    line: { color: COLORS[idx % COLORS.length], width: 1.5 },
  });

  const pairs: string[][] = [];
  for (let i = 0; i < perps.length; i += 2) pairs.push(perps.slice(i, i + 2));

  return (
    <div>
      {/* Row 1: portfolio equity | all-perp overlap */}
      <div style={GRID}>
        <Plot
          data={[{
            x: t,
            y: result.total_equity,
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio',
            line: { color: '#4C72B0', width: 2 },
          }]}
          layout={makeLayout('Cumulative Portfolio Equity', 'USD ($)')}
          config={CONFIG}
          style={{ ...PLOT_STYLE, height: '360px' }}
          useResizeHandler
        />
        <Plot
          data={perps.map((p, i) => perpTrace(p, i))}
          layout={makeLayout('Per-Asset PnL Contributions', 'PnL ($)', true)}
          config={CONFIG}
          style={{ ...PLOT_STYLE, height: '360px' }}
          useResizeHandler
        />
      </div>

      {/* Rows 2+: one chart per traded asset, 2 per row */}
      {pairs.map((pair) => (
        <div key={pair.join(',')} style={GRID}>
          {pair.map((p) => (
            <Plot
              key={p}
              data={[perpTrace(p, perps.indexOf(p))]}
              layout={makeLayout(p, 'PnL ($)')}
              config={CONFIG}
              style={{ ...PLOT_STYLE, height: '280px' }}
              useResizeHandler
            />
          ))}
          {pair.length === 1 && <div />}
        </div>
      ))}
    </div>
  );
}
