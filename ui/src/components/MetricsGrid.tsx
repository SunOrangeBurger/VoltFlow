"use client";

import { TrendingUp, TrendingDown, Zap, Gauge } from "lucide-react";
import type { TelemetryFrame } from "@/hooks/useSimulationSocket";

interface MetricsGridProps {
  frame: TelemetryFrame | null;
}

function Tile({
  label,
  value,
  unit,
  icon,
  tone = "default",
}: {
  label: string;
  value: string;
  unit?: string;
  icon: React.ReactNode;
  tone?: "default" | "positive" | "negative";
}) {
  const toneColor =
    tone === "positive" ? "text-volt-green" : tone === "negative" ? "text-volt-red" : "text-substation-text";

  return (
    <div className="rounded-lg border border-substation-border bg-substation-panel p-4">
      <div className="flex items-center justify-between text-substation-muted">
        <span className="text-xs">{label}</span>
        {icon}
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold ${toneColor}`}>
        {value}
        {unit && <span className="ml-1 text-sm text-substation-muted">{unit}</span>}
      </div>
    </div>
  );
}

export function MetricsGrid({ frame }: MetricsGridProps) {
  const pnl = frame?.cumulative_pnl ?? 0;
  const price = frame?.price ?? 0;

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Tile
        label="Cumulative PnL"
        value={pnl.toFixed(2)}
        unit="$"
        tone={pnl >= 0 ? "positive" : "negative"}
        icon={pnl >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
      />
      <Tile
        label="Spot Price"
        value={price.toFixed(2)}
        unit="$/MWh"
        icon={<Zap size={16} />}
      />
      <Tile
        label="Step Reward"
        value={(frame?.reward ?? 0).toFixed(4)}
        icon={<Gauge size={16} />}
        tone={(frame?.reward ?? 0) >= 0 ? "positive" : "negative"}
      />
      <Tile
        label="Degradation Cost"
        value={(frame?.degradation_cost ?? 0).toFixed(4)}
        unit="$"
        icon={<TrendingDown size={16} />}
        tone="negative"
      />
    </div>
  );
}
