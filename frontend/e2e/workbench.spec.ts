import { expect, test, type Page } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const screenshotDir = "/tmp/lakes-e2e-screenshots";

async function observePageErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

async function expectUsableMap(page: Page) {
  const map = page.locator(".ol-viewport").first();
  await expect(map).toBeVisible();
  const box = await map.boundingBox();
  expect(box?.width).toBeGreaterThan(250);
  expect(box?.height).toBeGreaterThan(250);
  await expect.poll(() => page.evaluate(() => performance.getEntriesByType("resource").some((entry) => entry.name.includes("/tiles/"))), { timeout: 20_000 }).toBe(true);
  await expect.poll(() => page.locator(".ol-layer canvas").count()).toBeGreaterThan(0);
  const canvas = page.locator(".ol-layer canvas").first();
  await expect.poll(() => canvas.evaluate((element) => {
    const context = element.getContext("2d");
    if (!context || !element.width || !element.height) return 0;
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let visible = 0;
    for (let index = 0; index < pixels.length; index += 40) {
      if (pixels[index + 3] > 0 && pixels[index] + pixels[index + 1] + pixels[index + 2] > 90) visible += 1;
    }
    return visible;
  }), { timeout: 20_000 }).toBeGreaterThan(500);
  await page.waitForTimeout(500);
}

test.beforeAll(async () => mkdir(screenshotDir, { recursive: true }));

test("user selection is the only unscoped entry", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/");
  await expect(page.getByText("选择一个用户工作空间")).toBeVisible();
  await expect(page.getByRole("button", { name: "新建用户" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "用户" })).toHaveCount(0);
  await page.screenshot({ path: `${screenshotDir}/profile-home-desktop.png`, fullPage: true });
  await page.getByRole("button", { name: "新建用户" }).click();
  await expect(page.getByRole("dialog", { name: "新建用户" })).toBeVisible();
  await expect(page.getByRole("button", { name: "空白" })).toBeVisible();
  await expect(page.getByRole("button", { name: "并集" })).toBeVisible();
  await page.getByRole("button", { name: "取消" }).click();
  await page.getByRole("button", { name: "进入用户 默认" }).click();
  await expect(page).toHaveURL(/#\/profiles\/default\/regions\/[^/]+\/sites$/);
  await expect(page.getByText("当前用户").first()).toBeVisible();
  await expect(page.getByText("默认", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "切换用户" }).first().click();
  await expect(page).toHaveURL(/#\/$/);
  await page.goto("#/regions/gansu/sites/gansu_17407");
  await expect(page.getByText("页面地址无效")).toBeVisible();
  await expect(page.getByRole("button", { name: "返回用户选择" })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/");
  await expect(page.getByRole("button", { name: "进入用户 默认" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/profile-home-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("site browser restores search and renders all map layers", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/profiles/default/regions/gansu/sites/gansu_17407?q=17407&has_osm=true&has_tci=true");
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true }).last()).toBeVisible();
  await expect(page.getByRole("button", { name: /区域 17407（苏干湖附近） 逻辑 \d+/ })).toBeVisible();
  await expect(page.getByPlaceholder("区域 ID / 名称提示")).toHaveValue("17407");
  await expect(page).toHaveURL(/#\/profiles\/default\/regions\/gansu\/sites\/gansu_17407\?q=17407$/);
  await expect(page.getByRole("button", { name: "切换用户" })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "区域" })).toBeVisible();
  for (const label of ["覆盖面积", "名称提示", "影像", "OSM 标注", "HydroLAKES", "本地标注"]) {
    await expect(page.getByRole("combobox", { name: label, exact: true })).toHaveCount(0);
  }
  for (const label of ["影像", "Tile", "OSM", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).toBeVisible();
  }
  await expect(page.getByRole("checkbox", { name: "影像", exact: true })).toBeChecked();
  for (const label of ["Tile", "OSM", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).not.toBeChecked();
  }
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/lake-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("training sample, patch review, and training history views load", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/profiles/default/regions/all/training/samples");
  await expect(page.getByText(/\d+ 个样本/)).toBeVisible();
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true }).first()).toBeVisible();

  await page.getByRole("tab", { name: "逻辑 Patch" }).click();
  await expect(page.getByText(/逻辑 Patch 审核 · [1-9]\d*/)).toBeVisible();
  const preview = page.locator('button img[loading="lazy"]').first();
  await expect(preview).toBeVisible();
  await expect.poll(() => preview.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  await preview.click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("dialog").screenshot({ path: `${screenshotDir}/patch-preview-dialog.png` });
  await page.getByRole("button", { name: "关闭" }).click();

  const sourceTab = page.getByRole("tab", { name: /来源冲突/ });
  if (await sourceTab.count()) {
    await sourceTab.click();
    await expect(page.getByRole("button", { name: "确认来源" }).first()).toBeVisible();
    await page.screenshot({ path: `${screenshotDir}/profile-source-conflicts.png`, fullPage: true });
  }

  await page.getByRole("tab", { name: "训练", exact: true }).click();
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  await expect(page.getByText("全部区域", { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: `${screenshotDir}/training-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("cached model validation deep link restores prediction", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/profiles/default/regions/shaanxi/model/shaanxi_23294?model=unet_current_v1%2Flast.pt");
  await expect(page.getByText(/区域 23294（喜河水库附近） · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("checkbox", { name: "影像", exact: true })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: "模型预测", exact: true })).toBeChecked();
  for (const label of ["Tile", "OSM", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).not.toBeChecked();
  }
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/model-validation-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("mobile site and training pages do not overflow", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/profiles/default/regions/gansu/sites/gansu_17407");
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true }).last()).toBeVisible();
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/lake-mobile.png`, fullPage: true });

  await page.goto("#/profiles/default/regions/all/training/patches");
  await expect(page.getByText(/逻辑 Patch 审核 · [1-9]\d*/)).toBeVisible();
  await expect(page.locator('button img[loading="lazy"]').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/patches-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("site map opens logical patch review without mutating data", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/profiles/default/regions/gansu/sites/gansu_17407");
  await expect(page.getByRole("checkbox", { name: "Patch", exact: true })).toBeVisible();
  await page.getByRole("checkbox", { name: "Patch", exact: true }).check();
  await expect(page.getByRole("combobox", { name: "训练样本 / 影像" })).toBeVisible();
  await expect(page.getByRole("button", { name: "排除" })).toBeVisible();
  await expect(page.getByText(/待处理 0/)).toBeVisible();
  await expect.poll(() => page.evaluate(() => performance.getEntriesByType("resource").some((entry) => entry.name.includes("/logical-patch-source/tiles/"))), { timeout: 30_000 }).toBe(true);
  const map = page.locator(".ol-viewport").first();
  await expect(map.locator(".ol-zoom-in")).toBeVisible();
  await expect(map.locator(".ol-zoom-out")).toBeVisible();
  await map.click({ position: { x: 500, y: 280 } });
  await expect(page.getByText(/gansu_17407_.*_lr\d+_lc\d+/)).toBeVisible({ timeout: 200 });
  await page.screenshot({ path: `${screenshotDir}/logical-patches-map.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("remaining mobile workspaces stay usable", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/profiles/default/regions/all/training/samples");
  await expect(page.getByText(/\d+ 个样本/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  await page.getByRole("tab", { name: "训练", exact: true }).click();
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/training-mobile.png`, fullPage: true });

  await page.goto("#/profiles/default/regions/shaanxi/model/shaanxi_23294?model=unet_current_v1%2Flast.pt");
  await expect(page.getByText(/区域 23294（喜河水库附近） · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/model-validation-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});
