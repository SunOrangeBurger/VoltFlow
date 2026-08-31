"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import type { TelemetryFrame } from "@/hooks/useSimulationSocket";

interface LiveChartProps {
  history: TelemetryFrame[];
  title: string;
  dataKey: keyof TelemetryFrame;
  color: string;
  unit?: string;
}

export function LiveChart({ history, title, dataKey, color, unit }: LiveChartProps) {
  return (
    <div className="rounded-lg border border-substation-border bg-substation-panel p-4">
      <h3 className="mb-3 text-sm font-medium text-substation-muted">{title}</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#232C36" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="step" stroke="#5C6B78" fontSize={11} tickLine={false} />
            <YAxis stroke="#5C6B78" fontSize={11} tickLine={false} />
            <Tooltip
              contentStyle={{
                background: "#12181F",
                border: "1px solid #232C36",
                borderRadius: 6,
                fontSize: 12,
              }}
              labelStyle={{ color: "#5C6B78" }}
              formatter={(value: number) => [`${value.toFixed(3)}${unit ? " " + unit : ""}`, title]}
            />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
