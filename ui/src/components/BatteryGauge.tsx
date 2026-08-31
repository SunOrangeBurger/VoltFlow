"use client";

interface BatteryGaugeProps {
  soc: number; // 0.0 - 1.0
  soh: number; // 0.0 - 1.0
  tCellC: number;
}

function arcPath(fraction: number, radius: number, cx: number, cy: number): string {
  const clamped = Math.max(0, Math.min(1, fraction));
  const startAngle = -220; // degrees, gauge sweep start
  const sweep = 260; // total gauge sweep in degrees
  const endAngle = startAngle + sweep * clamped;

  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const x1 = cx + radius * Math.cos(toRad(startAngle));
  const y1 = cy + radius * Math.sin(toRad(startAngle));
  const x2 = cx + radius * Math.cos(toRad(endAngle));
  const y2 = cy + radius * Math.sin(toRad(endAngle));
  const largeArc = sweep * clamped > 180 ? 1 : 0;

  return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
}

function trackPath(radius: number, cx: number, cy: number): string {
  const startAngle = -220;
  const sweep = 260;
  const endAngle = startAngle + sweep;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const x1 = cx + radius * Math.cos(toRad(startAngle));
  const y1 = cy + radius * Math.sin(toRad(startAngle));
  const x2 = cx + radius * Math.cos(toRad(endAngle));
  const y2 = cy + radius * Math.sin(toRad(endAngle));
  return `M ${x1} ${y1} A ${radius} ${radius} 0 1 1 ${x2} ${y2}`;
}

export function BatteryGauge({ soc, soh, tCellC }: BatteryGaugeProps) {
  const size = 220;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 85;

  const thermalWarn = tCellC > 40;
  const thermalCrit = tCellC > 45;

  return (
    <div className="rounded-lg border border-substation-border bg-substation-panel p-6">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="text-sm font-medium text-substation-muted">Cell State</h3>
        <span
          className={`font-mono text-xs ${
            thermalCrit ? "text-volt-red" : thermalWarn ? "text-volt-amber" : "text-substation-muted"
          }`}
        >
          {tCellC.toFixed(1)}&deg;C
        </span>
      </div>
      <div className="relative mx-auto" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <path
            d={trackPath(radius, cx, cy)}
            fill="none"
            stroke="#232C36"
            strokeWidth={14}
            strokeLinecap="round"
          />
          <path
            d={arcPath(soc, radius, cx, cy)}
            fill="none"
            stroke="#3DC9E8"
            strokeWidth={14}
            strokeLinecap="round"
          />
          <path
            d={arcPath(soh, radius - 20, cx, cy)}
            fill="none"
            stroke="#4CAF7D"
            strokeWidth={8}
            strokeLinecap="round"
            opacity={0.85}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl font-semibold text-substation-text">
            {(soc * 100).toFixed(1)}
            <span className="text-lg text-substation-muted">%</span>
          </span>
          <span className="text-xs text-substation-muted">State of Charge</span>
        </div>
      </div>
      <div className="mt-4 flex items-center justify-center gap-6 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-volt-cyan" />
          <span className="text-substation-muted">SoC</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-volt-green" />
          <span className="text-substation-muted">SoH {(soh * 100).toFixed(2)}%</span>
        </div>
      </div>
    </div>
  );
}
