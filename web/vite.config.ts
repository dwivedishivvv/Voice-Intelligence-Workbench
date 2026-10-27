import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      // shorthand string form doesn't upgrade websockets — /v1/ws/jobs/* needs ws:true
      // API_URL is set to http://api:8000 in Docker compose; falls back to localhost for native dev.
      "/v1": { target: process.env.API_URL ?? "http://localhost:8000", ws: true },
    },
  },
});
