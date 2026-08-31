"use client";

import { useEffect, useRef, useState } from "react";

export interface TelemetryFrame {
  step: number;
  soc: number;
  soh: number;
  t_cell_k: number;
  t_cell_c: number;
  price: number;
  revenue: number;
  degradation_cost: number;
  thermal_penalty: number;
  cumulative_pnl: number;
  reward: number;
}

interface SocketState {
  connected: boolean;
  latest: TelemetryFrame | null;
  history: TelemetryFrame[];
}

const HISTORY_LIMIT = 200;
const DEFAULT_WS_URL =
  process.env.NEXT_PUBLIC_VOLTFLOW_WS_URL ?? "ws://localhost:8000/ws/telemetry";

export function useSimulationSocket(url: string = DEFAULT_WS_URL): SocketState {
  const [connected, setConnected] = useState(false);
  const [latest, setLatest] = useState<TelemetryFrame | null>(null);
  const [history, setHistory] = useState<TelemetryFrame[]>([]);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let cancelled = false;

    const connect = () => {
      ws = new WebSocket(url);

      ws.onopen = () => {
        if (!cancelled) setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const frame: TelemetryFrame = JSON.parse(event.data);
          setLatest(frame);
          setHistory((prev) => {
            const next = [...prev, frame];
            return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next;
          });
        } catch {
          // Ignore malformed frames rather than crash the dashboard.
        }
      };

      ws.onclose = () => {
        if (!cancelled) {
          setConnected(false);
          reconnectTimer.current = setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws?.close();
    };
  }, [url]);

  return { connected, latest, history };
}
