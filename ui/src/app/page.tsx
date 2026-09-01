"use client";

import { useSimulationSocket } from "@/hooks/useSimulationSocket";
import { BatteryGauge } from "@/components/BatteryGauge";
import { MetricsGrid } from "@/components/MetricsGrid";
import { LiveChart } from "@/components/LiveChart";
import { ComparisonPanel } from "@/components/ComparisonPanel";
import { DayNightClock } from "@/components/DayNightClock";

export default function DashboardPage() {
  const { connected, latest, history } = useSimulationSocket();

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8 flex items-center justify-between border-b border-substation-border pb-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">VoltFlow</h1>
          <p className="text-sm text-substation-muted">
            Autonomous BESS arbitrage &amp; degradation dispatch — live telemetry
          </p>
          {latest?.policy_label && (
            <p className="mt-1 text-xs text-substation-muted">
              Driving policy:{" "}
              <span className="font-mono text-substation-text">{latest.policy_label}</span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${connected ? "bg-volt-green" : "bg-volt-red"}`}
          />
          <span className="text-xs text-substation-muted">
            {connected ? "Connected" : "Reconnecting..."}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="flex flex-col gap-6 lg:col-span-1">
          <BatteryGauge
            soc={latest?.soc ?? 0.5}
            soh={latest?.soh ?? 1.0}
            tCellC={latest?.t_cell_c ?? 25.0}
          />
          <DayNightClock
            hourOfDay={latest?.hour_of_day ?? 12.0}
            timeOfDay={latest?.time_of_day ?? "--:-- --"}
          />
        </div>
        <div className="lg:col-span-2">
          <MetricsGrid frame={latest} />
        </div>
      </div>

      <div className="mt-6">
        <ComparisonPanel
          strategies={latest?.strategies ?? null}
          policyLabel={latest?.policy_label ?? null}
          liveImprovementPct={latest?.live_improvement_pct ?? null}
          startupSelection={latest?.startup_selection ?? null}
        />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <LiveChart
          history={history}
          title="Cumulative PnL (PPO)"
          dataKey="cumulative_pnl"
          color="#4CAF7D"
          unit="$"
        />
        <LiveChart
          history={history}
          title="Spot Price"
          dataKey="price"
          color="#E8A23D"
          unit="$/MWh"
        />
        <LiveChart
          history={history}
          title="State of Charge"
          dataKey="soc"
          color="#3DC9E8"
        />
        <LiveChart
          history={history}
          title="Cell Temperature"
          dataKey="t_cell_c"
          color="#D9534F"
          unit="°C"
        />
      </div>

      <footer className="mt-10 border-t border-substation-border pt-4 text-xs text-substation-muted">
        Start the telemetry backend with{" "}
        <code className="rounded bg-substation-panel px-1.5 py-0.5">
          uvicorn voltflow.server.app:app --port 8000 --app-dir python
        </code>
        . See <code className="rounded bg-substation-panel px-1.5 py-0.5">README.md</code> for
        the full startup sequence, or run{" "}
        <code className="rounded bg-substation-panel px-1.5 py-0.5">./start_dashboard.sh</code>.
      </footer>
    </main>
  );
}