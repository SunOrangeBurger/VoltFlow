"use client";

import { TrendingUp, TrendingDown, Trophy, Cpu } from "lucide-react";
import type { StrategyFrame, StartupSelection } from "@/hooks/useSimulationSocket";

interface ComparisonPanelProps {
  strategies: {
    ppo: StrategyFrame;
    threshold: StrategyFrame;
    tou: StrategyFrame;
  } | null;
  policyLabel: string | null;
  liveImprovementPct: number | null;
  startupSelection: StartupSelection | null;
}

const STRATEGY_META: Record<string, { label: string; color: string }> = {
  ppo: { label: "VoltFlow RL (PPO)", color: "#4CAF7D" },
  threshold: { label: "Threshold Rule Heuristic", color: "#3DC9E8" },
  tou: { label: "TOU Heuristic", color: "#E8A23D" },
};

function fmt(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function ComparisonPanel({
  strategies,
  policyLabel,
  liveImprovementPct,
  startupSelection,
}: ComparisonPanelProps) {
  const order: Array<"ppo" | "threshold" | "tou"> = ["ppo", "threshold", "tou"];

  return (
    <div className="rounded-lg border border-substation-border bg-substation-panel p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-substation-muted">
          Live Strategy Comparison — Net Revenue &amp; Cost
        </h3>
        {liveImprovementPct !== null && liveImprovementPct !== undefined && (
          <div
            className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${
              liveImprovementPct >= 0
                ? "bg-volt-green/10 text-volt-green"
                : "bg-volt-red/10 text-volt-red"
            }`}
          >
            {liveImprovementPct >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {liveImprovementPct >= 0 ? "+" : ""}
            {fmt(liveImprovementPct, 1)}% vs best heuristic (this episode)
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="text-left text-xs text-substation-muted">
              <th className="pb-2 pr-3 font-normal">Strategy</th>
              <th className="pb-2 pr-3 font-normal">Net Revenue ($)</th>
              <th className="pb-2 pr-3 font-normal">Degradation Cost ($)</th>
              <th className="pb-2 pr-3 font-normal">Net PnL ($)</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {order.map((key) => {
              const s = strategies?.[key];
              const meta = STRATEGY_META[key];
              const isPpo = key === "ppo";
              return (
                <tr key={key} className="border-t border-substation-border">
                  <td className="py-2 pr-3 font-sans">
                    <span className="inline-flex items-center gap-1.5">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: meta.color }}
                      />
                      {isPpo ? policyLabel || meta.label : meta.label}
                    </span>
                  </td>
                  <td className="py-2 pr-3">{fmt(s?.cumulative_revenue)}</td>
                  <td className="py-2 pr-3 text-volt-red">{fmt(s?.cumulative_degradation)}</td>
                  <td
                    className={`py-2 pr-3 font-semibold ${
                      (s?.cumulative_pnl ?? 0) >= 0 ? "text-volt-green" : "text-volt-red"
                    }`}
                  >
                    {fmt(s?.cumulative_pnl)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {startupSelection && (
        <div className="mt-4 border-t border-substation-border pt-3">
          <div className="flex items-center gap-1.5 text-xs text-substation-muted">
            <Trophy size={12} />
            <span>
              Startup model selection winner:{" "}
              <span className="text-substation-text">
                {startupSelection.winner_label ?? "none (idle policy)"}
              </span>
              {startupSelection.winner_improvement_pct !== null && (
                <>
                  {" "}
                  (offline benchmark:{" "}
                  <span className="text-volt-green">
                    {startupSelection.winner_improvement_pct >= 0 ? "+" : ""}
                    {fmt(startupSelection.winner_improvement_pct, 1)}%
                  </span>{" "}
                  vs best heuristic)
                </>
              )}
            </span>
          </div>
          <details className="mt-2">
            <summary className="cursor-pointer text-xs text-substation-muted hover:text-substation-text">
              <span className="inline-flex items-center gap-1">
                <Cpu size={12} />
                {startupSelection.candidates.length} checkpoint(s) benchmarked at startup
              </span>
            </summary>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[480px] text-xs">
                <thead>
                  <tr className="text-left text-substation-muted">
                    <th className="pb-1 pr-3 font-normal">Checkpoint</th>
                    <th className="pb-1 pr-3 font-normal">Net PnL ($)</th>
                    <th className="pb-1 pr-3 font-normal">vs Heuristic</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {startupSelection.candidates.map((c) => (
                    <tr key={c.path} className="border-t border-substation-border/60">
                      <td className="py-1 pr-3 font-sans">
                        {c.label}
                        {c.path === startupSelection.winner && (
                          <span className="ml-1.5 rounded bg-volt-green/10 px-1 py-0.5 text-[10px] text-volt-green">
                            winner
                          </span>
                        )}
                      </td>
                      <td className="py-1 pr-3">
                        {c.error ? (
                          <span className="text-volt-red">error</span>
                        ) : (
                          `${fmt(c.net_pnl_mean)} ± ${fmt(c.net_pnl_std)}`
                        )}
                      </td>
                      <td className="py-1 pr-3">
                        {c.error
                          ? c.error
                          : `${c.improvement_pct >= 0 ? "+" : ""}${fmt(c.improvement_pct, 1)}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}