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
      "/v1": { target: "http://localhost:8000", ws: true },
    },
  },
});
