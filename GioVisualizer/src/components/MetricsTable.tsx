import { Table, Text } from '@mantine/core';
import type { BacktestResult, Metrics } from '../types';

const fmtNum = (v: number) => v.toFixed(4);
const fmtPct = (v: number) => `${(v * 100).toFixed(2)}%`;

interface Props {
  result: BacktestResult;
}

export default function MetricsTable({ result }: Props) {
  const perps = Object.keys(result.metrics_per_perp).sort();

  const row = (label: string, m: Metrics, bold = false) => (
    <Table.Tr key={label}>
      <Table.Td><Text fw={bold ? 700 : 400}>{label}</Text></Table.Td>
      <Table.Td><Text fw={bold ? 700 : 400}>{fmtNum(m.PnL)}</Text></Table.Td>
      <Table.Td><Text fw={bold ? 700 : 400}>{fmtPct(m.DD)}</Text></Table.Td>
      <Table.Td><Text fw={bold ? 700 : 400}>{fmtNum(m.Sharpe)}</Text></Table.Td>
      <Table.Td><Text fw={bold ? 700 : 400}>{fmtNum(m.Sortino)}</Text></Table.Td>
    </Table.Tr>
  );

  return (
    <Table mb="xl" withTableBorder withColumnBorders>
      <Table.Thead>
        <Table.Tr>
          <Table.Th>Asset</Table.Th>
          <Table.Th>PnL ($)</Table.Th>
          <Table.Th>Max DD</Table.Th>
          <Table.Th>Sharpe</Table.Th>
          <Table.Th>Sortino</Table.Th>
        </Table.Tr>
      </Table.Thead>
      <Table.Tbody>
        {perps.map((p) => row(p, result.metrics_per_perp[p]))}
        {row('TOTAL', result.metrics_total, true)}
      </Table.Tbody>
    </Table>
  );
}
