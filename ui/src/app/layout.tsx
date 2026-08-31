import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoltFlow — Live Dispatch Telemetry",
  description: "Autonomous BESS arbitrage & degradation management dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-substation-bg text-substation-text font-display antialiased">
        {children}
      </body>
    </html>
  );
}
