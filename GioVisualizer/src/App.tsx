import { useState } from "react";
import { Text, Loader } from "@mantine/core";
import type { BacktestResult } from "./types";
import DropZone from "./components/DropZone";
import Dashboard from "./components/Dashboard";
import { useTheme } from "./context/ThemeContext";
import logo from "./utils/logo.png";
import s from "./App.module.css";

export default function App() {
  const { isDark, toggle } = useTheme();
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFile = (file: File) => {
    setError(null);
    setIsLoading(true);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = JSON.parse(e.target?.result as string) as BacktestResult;
        if (
          !parsed.strategy ||
          !Array.isArray(parsed.timeline) ||
          !Array.isArray(parsed.total_equity)
        ) {
          throw new Error(
            "Invalid backtest JSON: missing strategy, timeline, or total_equity.",
          );
        }
        if (!parsed.per_perp_position) parsed.per_perp_position = {};
        if (!parsed.liquidation_events) parsed.liquidation_events = [];
        setResult(parsed);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsText(file);
  };

  if (!result) {
    return (
      <div className={s.landingWrap}>
        <div className={s.landingCard}>
          {isLoading ? (
            <div className={s.loadingWrap}>
              <img src={logo} alt="GioVisualizer" className={s.logo} />
              <Loader />
              <Text size="sm" style={{ color: "var(--text-secondary)" }}>
                Parsing backtest results...
              </Text>
            </div>
          ) : (
            <>
              <img src={logo} alt="GioVisualizer" className={s.logoLanding} />
              <Text fw={600} size="xl" style={{ color: "var(--text-primary)" }}>
                GioVisualizer
              </Text>
              <Text
                size="sm"
                mt={4}
                mb="xl"
                style={{ color: "var(--text-secondary)" }}
              >
                Drop a backtest JSON to visualize results.
              </Text>
              <DropZone onFile={handleFile} />
              {error && <div className={s.errorText}>{error}</div>}
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={s.page}>
      <header className={s.header}>
        <div className={s.headerLeft}>
          <img src={logo} alt="GioVisualizer" className={s.logoSmall} />
          <Text fw={600} size="md" style={{ color: "var(--text-primary)" }}>
            GioVisualizer
          </Text>
        </div>
        <div className={s.headerRight}>
          <button onClick={toggle} className={s.themeBtn}>
            {isDark ? "☀" : "🌙"}
          </button>
          <button className={s.loadBtn} onClick={() => setResult(null)}>
            Load another
          </button>
        </div>
      </header>
      <main className={s.content}>
        <Dashboard result={result} />
      </main>
    </div>
  );
}
