import { test, expect } from "@playwright/test";

test("random selection shows map before prediction and clears mask on failure", async ({ page }) => {
  const workspace = { id: "default", name: "默认", status: "active", selected_patch_count: 0, site_count: 0, default: true };
  const user = { id: "default", name: "Test", status: "active", default_workspace_id: "default", workspace, default: true, role: "admin" };
  const site = { site_id: "hunan_test", region: "hunan", display_name: "地图先加载测试区域", coverage_area_km2: 1, bbox: [112, 28, 112.05, 28.05], geometry: { type: "Polygon", coordinates: [[[112,28],[112.05,28],[112.05,28.05],[112,28.05],[112,28]]] } };
  const model = { key: "workspaces/default/gansu/test/best.pt", label: "Test model", name: "Test model", scope: "gansu", in_channels: 5, config: { input_bands: ["blue", "green", "red", "nir", "swir1"] } };
  const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aTioAAAAASUVORK5CYII=", "base64");
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  let releaseTiles!: () => void;
  const tilesPending = new Promise<void>((resolve) => { releaseTiles = resolve; });
  let tileRequests = 0;
  let predictionRequests = 0;
  let oldRandomRequests = 0;
  const hunanListRequests: string[] = [];
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (!url.hostname.match(/^(127[.]0[.]0[.]1|localhost)$/)) return route.abort();
    if (url.pathname.includes("/tiles/") || url.pathname.startsWith("/mock-tiles/")) {
      tileRequests++;
      await tilesPending;
      return route.fulfill({ contentType: "image/png", body: pixel });
    }
    if (!url.pathname.startsWith("/api/")) return route.continue();
    if (url.pathname === "/api/auth/session") return route.fulfill({ json: { authenticated: true, mode: "development", identity: {}, user } });
    if (url.pathname === "/api/workspaces/default") return route.fulfill({ json: { workspace } });
    if (url.pathname === "/api/regions") return route.fulfill({ json: { default: "gansu", items: [{ key: "gansu", name: "甘肃", ready: true, site_count: 1 }, { key: "hunan", name: "湖南", ready: true, site_count: 1 }] } });
    if (url.pathname === "/api/workspaces/default/regions/hunan/sites") {
      hunanListRequests.push(url.searchParams.get("q") || "");
      return route.fulfill({ json: { total: 1, items: [site] } });
    }
    if (url.pathname.endsWith("/model-validation/models")) return route.fulfill({ json: { default: model.key, items: [model] } });
    if (url.pathname.endsWith("/model-validation/random-target")) {
      expect(url.pathname).toContain("/regions/all/");
      return route.fulfill({ json: { region: "hunan", site_id: site.site_id, site } });
    }
    if (url.pathname.endsWith("/model-validation/random")) { oldRandomRequests++; return route.fulfill({ status: 500, json: { error: "Old combined endpoint must not be called" } }); }
    if (url.pathname.endsWith("/model-prediction")) {
      predictionRequests++;
      if (predictionRequests > 1) return route.fulfill({ status: 500, json: { error: "模拟推理失败" } });
      await pending;
      return route.fulfill({ json: { region: "hunan", site_id: site.site_id, site, model: { key: model.key, name: model.name }, prediction: { type: "FeatureCollection", features: [{ type: "Feature", geometry: site.geometry, properties: {} }] }, stats: { area_km2: 1, predicted_ratio: .25 } } });
    }
    if (url.pathname.endsWith("/tile-meta")) return route.fulfill({ json: { tile_url: "/mock-tiles/{z}/{x}/{y}.png", site_bounds: site.bbox, tile_bounds: site.bbox } });
    if (url.pathname.endsWith("/sites/hunan_test")) return route.fulfill({ json: site });
    if (url.pathname.endsWith("/sentinel-tiles")) return route.fulfill({ json: { tiles: [] } });
    if (url.pathname.endsWith("/context-water")) return route.fulfill({ json: { min_area_km2: 10, sources: { osm: { type: "FeatureCollection", features: [] }, hydrolakes: { type: "FeatureCollection", features: [] } } } });
    return route.fulfill({ json: { total: 0, items: [], assets: [], runs: [], facets: {}, type: "FeatureCollection", features: [] } });
  });
  await page.goto("#/users/default/workspaces/default/regions/gansu/models/validate", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "单模型地图", exact: true }).click();
  await page.getByRole("button", { name: "随机验证一个区域", exact: true }).click();
  await expect.poll(() => tileRequests).toBeGreaterThan(0);
  // Tile responses are still blocked: inference must start independently.
  await expect.poll(() => predictionRequests).toBe(1);
  releaseTiles();
  await expect(page.getByTestId("site-map")).toBeVisible();
  await expect(page.locator(".ol-viewport")).toBeVisible();
  await expect(page.getByText("地图已加载，正在生成预测结果…", { exact: true })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "模型预测", exact: true })).toHaveCount(0);
  expect(hunanListRequests).toEqual([]);
  const picker = page.getByRole("combobox", { name: "选择观测区域", exact: true });
  await expect(picker).toHaveValue(site.display_name);
  await picker.click();
  await expect.poll(() => hunanListRequests.length).toBeGreaterThan(0);
  await picker.fill("地图");
  await expect.poll(() => hunanListRequests.includes("地图")).toBe(true);
  await picker.press("Escape");
  await page.screenshot({ path: "/tmp/lakes-map-first-pending.png", fullPage: true });
  release();
  await expect(page.getByRole("checkbox", { name: "模型预测", exact: true })).toBeChecked();
  await expect(page.getByText(/水体像元 25[.]0%/).first()).toBeVisible();
  expect(oldRandomRequests).toBe(0);
  await page.screenshot({ path: "/tmp/lakes-map-first-success.png", fullPage: true });
  await page.getByRole("button", { name: "模型预测", exact: true }).click();
  await expect(page.getByText("模拟推理失败", { exact: true })).toBeVisible();
  await expect(page.getByTestId("site-map")).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "模型预测", exact: true })).toHaveCount(0);
});
