/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        substation: {
          bg: "#0A0E12",
          panel: "#12181F",
          border: "#232C36",
          text: "#D8E1E8",
          muted: "#5C6B78",
        },
        volt: {
          amber: "#E8A23D",
          cyan: "#3DC9E8",
          green: "#4CAF7D",
          red: "#D9534F",
        },
      },
      fontFamily: {
        display: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
