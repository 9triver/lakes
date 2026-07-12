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
  }), { timeout: 20_000 }).toBeGreaterThan(2_000);
  await page.waitForTimeout(500);
}

test.beforeAll(async () => mkdir(screenshotDir, { recursive: true }));

test("legacy frontend remains available", async ({ page }) => {
  await page.goto("legacy/");
  await expect(page.locator("#region-select")).toBeVisible();
  await expect(page.locator("#lake-list")).toBeVisible();
});

test("lake browser restores filters and renders all map layers", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/regions/gansu/lakes/gansu_17407?water_type=lake&has_tci=true");
  await expect(page.getByText("苏干湖", { exact: true }).last()).toBeVisible();
  await expect(page.getByRole("combobox", { name: "类型" })).toHaveText(/湖泊/);
  await expect(page.getByRole("combobox", { name: "影像" })).toHaveText(/有影像/);
  for (const label of ["影像", "Tile", "OSM", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).toBeVisible();
  }
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/lake-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("training sample, patch review, and training history views load", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/regions/all/training/samples");
  await expect(page.getByText(/\d+ 个样本/)).toBeVisible();
  await expect(page.getByText("苏干湖", { exact: true }).first()).toBeVisible();

  await page.getByRole("tab", { name: "Patch 审核" }).click();
  await expect(page.getByText(/Patch 审核 · [1-9]\d*/)).toBeVisible();
  const preview = page.locator('button img[loading="lazy"]').first();
  await expect(preview).toBeVisible();
  await expect.poll(() => preview.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  await preview.click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("dialog").screenshot({ path: `${screenshotDir}/patch-preview-dialog.png` });
  await page.getByRole("button", { name: "关闭" }).click();

  await page.getByRole("tab", { name: "训练", exact: true }).click();
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  await expect(page.getByText("全部区域", { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: `${screenshotDir}/training-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("cached model validation deep link restores prediction", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/regions/shaanxi/model/shaanxi_23294?model=unet_current_v1%2Flast.pt");
  await expect(page.getByText(/喜河水库 · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByLabel("模型预测", { exact: true })).toBeVisible();
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/model-validation-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("mobile lake and training pages do not overflow", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/regions/gansu/lakes/gansu_17407");
  await expect(page.getByText("苏干湖", { exact: true }).last()).toBeVisible();
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/lake-mobile.png`, fullPage: true });

  await page.goto("#/regions/all/training/patches");
  await expect(page.getByText(/Patch 审核 · [1-9]\d*/)).toBeVisible();
  await expect(page.locator('button img[loading="lazy"]').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/patches-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("remaining mobile workspaces stay usable", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/regions/all/training/samples");
  await expect(page.getByText(/\d+ 个样本/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  await page.getByRole("tab", { name: "训练", exact: true }).click();
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/training-mobile.png`, fullPage: true });

  await page.goto("#/regions/shaanxi/model/shaanxi_23294?model=unet_current_v1%2Flast.pt");
  await expect(page.getByText(/喜河水库 · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/model-validation-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});
