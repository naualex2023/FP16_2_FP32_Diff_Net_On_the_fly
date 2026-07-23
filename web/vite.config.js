import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend port is configurable via SD_API_PORT (default 8765).
const apiPort = process.env.SD_API_PORT || "8765";
const apiTarget = `http://localhost:${apiPort}`;

// Dev server runs on 5174 (5173 may conflict with other tools).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5174,
    proxy: {
      "/api": apiTarget,
      "/gallery": apiTarget,
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});