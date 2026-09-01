"use client";

import { useMemo } from "react";
import { Sun, Moon, Sunrise, Sunset } from "lucide-react";

interface DayNightClockProps {
  hourOfDay: number; // 0.0 - 24.0, fractional
  timeOfDay: string; // pre-formatted "HH:MM AM/PM" from the backend
}

// Boundaries for the day/night cycle (24h clock, fractional hours).
const SUNRISE_START = 5.0;
const SUNRISE_END = 7.5;
const SUNSET_START = 17.5;
const SUNSET_END = 20.0;

type Phase = "night" | "sunrise" | "day" | "sunset";

function phaseAt(hour: number): Phase {
  if (hour >= SUNRISE_START && hour < SUNRISE_END) return "sunrise";
  if (hour >= SUNRISE_END && hour < SUNSET_START) return "day";
  if (hour >= SUNSET_START && hour < SUNSET_END) return "sunset";
  return "night";
}

const PHASE_META: Record<Phase, { label: string; sky: [string, string]; icon: React.ReactNode }> = {
  night: { label: "Night", sky: ["#0A0E12", "#161D26"], icon: <Moon size={14} /> },
  sunrise: { label: "Sunrise", sky: ["#1F2A38", "#E8A23D"], icon: <Sunrise size={14} /> },
  day: { label: "Day", sky: ["#1B4965", "#3DC9E8"], icon: <Sun size={14} /> },
  sunset: { label: "Sunset", sky: ["#3D2A1F", "#D9534F"], icon: <Sunset size={14} /> },
};

/** Maps hour-of-day to a 0..1 arc position, where 0 = horizon (sunrise),
 * 0.5 = zenith (noon), 1 = horizon (sunset), wrapping back down through
 * the night on the underside of the same arc so the sun/moon always
 * traces one continuous loop across 24h. */
function arcPosition(hour: number): { x: number; y: number; aboveHorizon: boolean } {
  // Treat SUNRISE_START..SUNSET_END as the "above horizon" half of the
  // loop, and the remaining night hours as the "below horizon" half —
  // each mapped independently onto a semicircle so the sun/moon glides
  // smoothly rather than jumping at the phase boundaries.
  const dayLen = SUNSET_END - SUNRISE_START; // hours the sun is above the horizon
  const nightLen = 24 - dayLen;

  let angle: number; // 0 = left horizon, PI = right horizon, going over the top
  let aboveHorizon: boolean;

  if (hour >= SUNRISE_START && hour <= SUNSET_END) {
    const frac = (hour - SUNRISE_START) / dayLen; // 0..1
    angle = Math.PI * (1 - frac); // PI (left) -> 0 (right), arcing over top
    aboveHorizon = true;
  } else {
    const wrapped = hour < SUNRISE_START ? hour + 24 : hour;
    const frac = (wrapped - SUNSET_END) / nightLen; // 0..1
    angle = Math.PI * frac; // 0 (right) -> PI (left), arcing under bottom
    aboveHorizon = false;
  }

  const cx = 100;
  const cy = 60; // horizon line
  const r = 46;
  const x = cx - r * Math.cos(angle);
  // Above horizon arcs upward (y decreases); below horizon arcs downward
  // (y increases) — both driven by the magnitude of sin(angle).
  const y = aboveHorizon ? cy - r * Math.sin(angle) : cy + r * Math.sin(angle);

  return { x, y, aboveHorizon };
}

export function DayNightClock({ hourOfDay, timeOfDay }: DayNightClockProps) {
  const phase = phaseAt(hourOfDay);
  const meta = PHASE_META[phase];

  const { x, y, aboveHorizon } = useMemo(() => arcPosition(hourOfDay), [hourOfDay]);

  const width = 200;
  const height = 100;
  const horizonY = 60;

  return (
    <div className="rounded-lg border border-substation-border bg-substation-panel p-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-medium text-substation-muted">Time of Day</h3>
        <span className="flex items-center gap-1 text-xs text-substation-muted">
          {meta.icon}
          {meta.label}
        </span>
      </div>

      <div className="relative overflow-hidden rounded-md" style={{ height }}>
        <svg
          width="100%"
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          className="absolute inset-0"
        >
          <defs>
            <linearGradient id="sky-gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={meta.sky[0]} />
              <stop offset="100%" stopColor={meta.sky[1]} />
            </linearGradient>
          </defs>
          <rect
            x={0}
            y={0}
            width={width}
            height={height}
            fill="url(#sky-gradient)"
            style={{ transition: "fill 1s ease" }}
          />
          {/* faint arc path the sun/moon travels along */}
          <path
            d={`M 54 ${horizonY} A 46 46 0 0 1 146 ${horizonY}`}
            fill="none"
            stroke="#ffffff"
            strokeOpacity={0.12}
            strokeWidth={1}
            strokeDasharray="2 3"
          />
          {/* horizon line */}
          <line
            x1={0}
            y1={horizonY}
            x2={width}
            y2={horizonY}
            stroke="#5C6B78"
            strokeOpacity={0.4}
            strokeWidth={1}
          />
          {/* ground */}
          <rect x={0} y={horizonY} width={width} height={height - horizonY} fill="#12181F" fillOpacity={0.7} />

          {/* sun or moon, gliding along the arc */}
          <g style={{ transition: "transform 0.6s linear" }} transform={`translate(${x}, ${y})`}>
            {aboveHorizon ? (
              <circle r={7} fill="#E8A23D" style={{ filter: "drop-shadow(0 0 6px #E8A23Daa)" }} />
            ) : (
              <circle r={5.5} fill="#D8E1E8" fillOpacity={0.85} />
            )}
          </g>
        </svg>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="font-mono text-2xl font-semibold text-substation-text">{timeOfDay}</span>
        <span className="font-mono text-xs text-substation-muted">
          hour {hourOfDay.toFixed(2)} / 24
        </span>
      </div>
    </div>
  );
}