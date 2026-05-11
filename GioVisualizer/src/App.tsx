import { useState } from 'react';
import { Button, Container, Group, Text, Title } from '@mantine/core';
import type { BacktestResult } from './types';
import Charts from './components/Charts';
import DropZone from './components/DropZone';
import MetricsTable from './components/MetricsTable';

export default function App() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File) => {
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = JSON.parse(e.target?.result as string) as BacktestResult;
        if (!parsed.strategy || !Array.isArray(parsed.timeline) || !Array.isArray(parsed.total_equity)) {
          throw new Error('Invalid backtest JSON: missing strategy, timeline, or total_equity.');
        }
        setResult(parsed);
      } catch (err) {
        setError((err as Error).message);
      }
    };
    reader.readAsText(file);
  };

  if (!result) {
    return (
      <Container size="sm" py="xl">
        <Title order={1} mb="xs">GioVisualizer</Title>
        <Text c="dimmed" mb="xl">Drop a backtest JSON to visualize results.</Text>
        <DropZone onFile={handleFile} />
        {error && <Text c="red" mt="md">{error}</Text>}
      </Container>
    );
  }

  return (
    <Container size="xl" py="xl">
      <Group justify="space-between" mb="xl">
        <div>
          <Title order={1}>{result.strategy}</Title>
          <Text c="dimmed">
            {result.timeline.length} bars &middot; {result.n_opened} opened &middot;{' '}
            {result.n_closed} closed &middot; {result.n_liquidated} liquidated
          </Text>
        </div>
        <Button variant="subtle" onClick={() => setResult(null)}>Load another</Button>
      </Group>
      <MetricsTable result={result} />
      <Charts result={result} />
    </Container>
  );
}
