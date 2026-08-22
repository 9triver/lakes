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

async function expectToolbarOutsideMap(page: Page) {
  const toolbar = page.getByTestId("site-map-toolbar");
  const map = page.getByTestId("site-map");
  await expect(toolbar).toBeVisible();
  await expect(map).toBeVisible();
  const [toolbarBox, mapBox] = await Promise.all([toolbar.boundingBox(), map.boundingBox()]);
  expect(toolbarBox).not.toBeNull();
  expect(mapBox).not.toBeNull();
  expect((toolbarBox?.y || 0) + (toolbarBox?.height || 0)).toBeLessThanOrEqual((mapBox?.y || 0) + 1);
}

async function expectSidebarUserVisible(page: Page) {
  const aside = page.locator("aside");
  const user = page.getByTestId("sidebar-user");
  await expect(user).toBeVisible();
  const [asideBox, userBox] = await Promise.all([aside.boundingBox(), user.boundingBox()]);
  expect(asideBox).not.toBeNull();
  expect(userBox).not.toBeNull();
  expect(userBox!.y).toBeGreaterThanOrEqual(asideBox!.y);
  expect(userBox!.y + userBox!.height).toBeLessThanOrEqual(asideBox!.y + asideBox!.height + 1);
}

async function expectHorizontalWorkspaceNavigation(page: Page) {
  const navigation = page.getByRole("navigation", { name: "工作模式" });
  const buttons = navigation.getByRole("button");
  await expect(buttons).toHaveCount(3);
  const boxes = await buttons.evaluateAll((items) => items.map((item) => item.getBoundingClientRect().toJSON()));
  expect(Math.max(...boxes.map((box) => box.y)) - Math.min(...boxes.map((box) => box.y))).toBeLessThan(2);
}

test.beforeAll(async () => mkdir(screenshotDir, { recursive: true }));

test("authenticated user is routed to the assigned workspace", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/");
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/[^/]+\/sites$/);
  await expect(page.getByText("当前用户").first()).toBeVisible();
  await expect(page.getByText("湖泊工作台", { exact: true })).toBeVisible();
  await expect(page.getByText("观测与训练工作台", { exact: true })).toHaveCount(0);
  await expect(page.getByText("工作区", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "模型实验", exact: true })).toBeVisible();
  await expectHorizontalWorkspaceNavigation(page);
  await expect(page.getByText("caochun@gmail.com", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "新建用户" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "切换用户" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "退出" })).toHaveCount(0);
  await page.screenshot({ path: `${screenshotDir}/workspace-home-desktop.png`, fullPage: true });
  await page.goto("/?__cf_access_message=logged_out#/");
  await expect.poll(() => page.evaluate(() => window.location.search)).toBe("");
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/[^/]+\/sites$/);
  await page.goto("#/regions/gansu/sites/gansu_17407");
  await expect(page.getByText("页面地址无效")).toBeVisible();
  await page.getByRole("button", { name: "返回当前工作区" }).click();
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/[^/]+\/sites$/);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/");
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/[^/]+\/sites$/);
  await expect(page.getByText("当前用户").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/workspace-home-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("site browser filters by region and renders all map layers", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/users/default/workspaces/default/regions/gansu/sites/gansu_17407");
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true }).last()).toBeVisible();
  await expectSidebarUserVisible(page);
  await expect(page.getByRole("button", { name: /区域 17407（苏干湖附近）/ }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "切换用户" })).toHaveCount(0);
  await expect(page.getByRole("combobox", { name: "区域" })).toContainText("甘肃省 (19)");
  for (const label of ["覆盖面积", "名称提示", "影像", "OSM 标注", "HydroLAKES", "本地标注"]) {
    await expect(page.getByRole("combobox", { name: label, exact: true })).toHaveCount(0);
  }
  await expect(page.getByRole("combobox", { name: "底图" })).toContainText("卫星图");
  await expect(page.getByRole("combobox", { name: "本地影像期次" })).toHaveCount(1);
  await expect(page.getByTestId("site-map-toolbar").getByRole("combobox", { name: "本地影像期次" })).toBeVisible();
  await expect(page.getByText(/29 期本地影像/).first()).toBeVisible();
  await expect(page.getByText(/最近影像 2025-07-31/).first()).toBeVisible();
  await page.getByRole("button", { name: "收起侧栏" }).click();
  await expect(page.getByRole("button", { name: "展开侧栏" })).toBeVisible();
  await page.getByRole("button", { name: "展开侧栏" }).click();
  await expect(page.getByRole("button", { name: "收起侧栏" })).toBeVisible();
  for (const label of ["影像", "Tile", "OSM 水体", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).toBeVisible();
  }
  await expect(page.getByRole("checkbox", { name: "影像", exact: true })).toBeChecked();
  for (const label of ["Tile", "OSM 水体", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).not.toBeChecked();
  }
  await expectToolbarOutsideMap(page);
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/lake-desktop.png`, fullPage: true });
  await page.getByRole("combobox", { name: "区域" }).click();
  await page.getByRole("option", { name: "陕西省 (80)" }).click();
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/shaanxi\/sites$/);
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true })).toHaveCount(0);
  await expect(page.getByText(/^共 80 个/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("training data and model training views load", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/users/default/workspaces/default/regions/all/training-data");
  await expect(page.getByRole("heading", { name: "训练数据", exact: true })).toBeVisible();
  await expectSidebarUserVisible(page);
  const sidebar = page.locator("aside");
  await expect(sidebar.getByText(/^\d+ 个训练区域 · \d+ 个 Patch$/)).toBeVisible();
  await expect(sidebar.getByText("全部训练区域", { exact: true })).toBeVisible();
  const trainingSite = sidebar.getByRole("button").filter({ hasText: "区域 17407（苏干湖附近）" });
  await expect(trainingSite).toBeVisible();
  const sitePatchCount = Number((await trainingSite.getByText(/^\d+ 个 Patch/).textContent())?.match(/^\d+/)?.[0]);
  expect(sitePatchCount).toBeGreaterThan(0);
  await trainingSite.click();
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/all\/training-data\/gansu_17407$/);
  await expect(page.getByText(`全部 ${sitePatchCount}`, { exact: true })).toBeVisible();
  await sidebar.getByText("全部训练区域", { exact: true }).click();
  await expect(page).toHaveURL(/#\/users\/default\/workspaces\/default\/regions\/all\/training-data$/);
  await expect(page.getByText(/^全部 [1-9]\d*/)).toBeVisible();
  await expect(page.getByText(/^已纳入 [1-9]\d*/)).toBeVisible();
  await expect(page.locator('img[src*="/logical-patches/"]').first()).toBeVisible();
  await page.goto("#/users/default/workspaces/default/regions/all/models/train");
  await expect(page.getByRole("heading", { name: "模型", exact: true })).toBeVisible();
  await expect(page.getByRole("tab", { name: "训练", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("combobox", { name: "数据来源" })).toContainText("当前 Workspace");
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  await expectSidebarUserVisible(page);
  await page.screenshot({ path: `${screenshotDir}/training-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("cached model validation deep link restores prediction", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/users/default/workspaces/default/regions/shaanxi/models/validate/shaanxi_23294?model=workspaces%2Fdefault%2Fshaanxi%2Funet_current_v1%2Flast.pt");
  await expect(page.getByText(/区域 23294（喜河水库附近） · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("checkbox", { name: "影像", exact: true })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: "模型预测", exact: true })).toBeChecked();
  for (const label of ["Tile", "OSM 水体", "HydroLAKES", "其他", "ESA", "JRC", "本地标注"]) {
    await expect(page.getByRole("checkbox", { name: label, exact: true })).not.toBeChecked();
  }
  await expectToolbarOutsideMap(page);
  await expectUsableMap(page);
  await page.screenshot({ path: `${screenshotDir}/model-validation-desktop.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("mobile site and training pages do not overflow", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/users/default/workspaces/default/regions/gansu/sites/gansu_17407");
  await expect(page.getByText("区域 17407（苏干湖附近）", { exact: true }).last()).toBeVisible();
  await expectToolbarOutsideMap(page);
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/lake-mobile.png`, fullPage: true });

  await page.goto("#/users/default/workspaces/default/regions/all/training-data");
  await expect(page.getByRole("heading", { name: "训练数据", exact: true })).toBeVisible();
  await expect(page.locator('img[src*="/logical-patches/"]').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/patches-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("default workspace exposes its migrated logical patches", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.goto("#/users/default/workspaces/default/regions/gansu/sites/gansu_17407");
  await expectUsableMap(page);
  await expect(page.getByRole("checkbox", { name: "Patch", exact: true })).toHaveCount(1);
  await page.goto("#/users/default/workspaces/default/regions/gansu/training-data");
  await expect(page.getByRole("heading", { name: "训练数据", exact: true })).toBeVisible();
  await expect(page.locator('img[src*="/logical-patches/"]').first()).toBeVisible();
  await page.screenshot({ path: `${screenshotDir}/default-workspace-patches.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("remaining mobile workspaces stay usable", async ({ page }) => {
  const errors = await observePageErrors(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("#/users/default/workspaces/default/regions/all/training-data");
  await expect(page.getByRole("heading", { name: "训练数据", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  await page.goto("#/users/default/workspaces/default/regions/all/models/train");
  await expect(page.getByRole("tab", { name: "训练", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText(/历史任务 \(\d+\)/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/training-mobile.png`, fullPage: true });

  await page.goto("#/users/default/workspaces/default/regions/shaanxi/models/validate/shaanxi_23294?model=workspaces%2Fdefault%2Fshaanxi%2Funet_current_v1%2Flast.pt");
  await expect(page.getByText(/区域 23294（喜河水库附近） · 模型 unet_current_v1/)).toBeVisible({ timeout: 30_000 });
  await expectUsableMap(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: `${screenshotDir}/model-validation-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
});
