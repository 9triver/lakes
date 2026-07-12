import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/static/dist/",
  build: {
    outDir: "../src/lake_workbench/static/dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/ol/")) return "openlayers";
          if (id.includes("node_modules/@mui/") || id.includes("node_modules/@emotion/") || id.includes("node_modules/react") || id.includes("node_modules/scheduler")) return "ui";
          return undefined;
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:18765",
    },
  },
});
