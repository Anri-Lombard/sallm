import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const snapshotPath = path.resolve(__dirname, "../sallm_memory/observability/latest.json");

function snapshotRoute() {
  return {
    name: "sallm-snapshot-route",
    configureServer(server) {
      server.middlewares.use("/latest.json", (_req, res) => {
        fs.readFile(snapshotPath, "utf8", (err, body) => {
          if (err) {
            res.statusCode = 404;
            res.setHeader("content-type", "application/json");
            res.end(JSON.stringify({ error: "snapshot not found", path: snapshotPath }));
            return;
          }
          res.setHeader("content-type", "application/json");
          res.setHeader("cache-control", "no-store");
          res.end(body);
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), snapshotRoute()],
});
