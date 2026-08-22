import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  outputDir: "/tmp/lakes-playwright-results",
  timeout: 45_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: process.env.LAKES_E2E_BASE_URL || "http://127.0.0.1:18765/",
    trace: "retain-on-failure",
  },
});
