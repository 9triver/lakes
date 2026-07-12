import { escapeHtml, formatPercentText } from "./formatters.js";

export function filterPatches(patches, waterFilter) {
  return patches.filter((patch) => {
    if (waterFilter === "water" && Number(patch.water_pixels || 0) <= 0) return false;
    if (waterFilter === "negative" && Number(patch.water_pixels || 0) > 0) return false;
    return true;
  });
}

export function normalizePage(page, count, pageSize) {
  const pageCount = Math.max(1, Math.ceil(count / pageSize));
  return Math.min(Math.max(1, page), pageCount);
}

export function createPatchReviewController({
  state,
  elements,
  isAllRegions,
  updateTrainingPatch,
  setTrainingView,
  selectLake,
  showError,
}) {
  const { grid, summary, pageLabel, previous, next, waterFilter, modal, modalClose, modalTitle, modalSubtitle, modalImage, modalMeta, modalToggle } = elements;

  function filteredTrainingPatches() {
    return filterPatches(state.trainingPatches, state.patchWaterFilter);
  }

  function normalizePatchPage(count = filteredTrainingPatches().length) {
    state.patchPage = normalizePage(state.patchPage, count, state.patchPageSize);
  }

  function renderPatchReview() {
    grid.replaceChildren();
    const patches = filteredTrainingPatches();
    normalizePatchPage(patches.length);
    const pageCount = Math.max(1, Math.ceil(patches.length / state.patchPageSize));
    const start = (state.patchPage - 1) * state.patchPageSize;
    const pageItems = patches.slice(start, start + state.patchPageSize);
    summary.textContent = `${patches.length} 个 patch，当前第 ${state.patchPage} 页`;
    pageLabel.textContent = `${state.patchPage} / ${pageCount}`;
    previous.disabled = state.patchPage <= 1;
    next.disabled = state.patchPage >= pageCount;
    if (!pageItems.length) {
      const empty = document.createElement("div");
      empty.className = "patch-empty";
      empty.textContent = isAllRegions()
        ? "暂无 patch，请先生成 patch"
        : "当前区域暂无 patch，请先生成 patch，或切换到“全部”查看已有区域的 patch";
      grid.append(empty);
      return;
    }
    for (const patch of pageItems) {
      const item = document.createElement("article");
      item.className = `patch-card${patch.included ? "" : " excluded"}${patch.preview_exists ? "" : " missing"}`;
      const name = patch.lake_display_name || patch.lake_name || patch.lake_id || patch.sample_id;
      item.innerHTML = `
        <button class="patch-card-image" type="button" aria-label="打开 patch 预览">${patch.preview_url ? `<img src="${escapeHtml(patch.preview_url)}" alt="" loading="lazy" />` : ""}</button>
        <div class="patch-card-body">
          <div class="patch-card-top"><strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong><span class="badge">${patch.included ? "include" : "exclude"}</span></div>
          <div class="patch-card-meta">${escapeHtml(patch.patch_id || "")}</div>
          <div class="patch-card-stats"><span>water ${escapeHtml(formatPercentText(patch.water_ratio_valid))}</span><span>valid ${escapeHtml(formatPercentText(patch.valid_ratio))}</span><span>ignore ${escapeHtml(patch.ignore_pixels || 0)}</span></div>
        </div>`;
      item.querySelector(".patch-card-image").addEventListener("click", () => openPatchModal(patch));
      const actions = document.createElement("div");
      actions.className = "patch-card-actions";
      const include = document.createElement("button");
      include.type = "button";
      include.textContent = patch.included ? "排除" : "恢复包含";
      include.addEventListener("click", () => updateTrainingPatch(patch.patch_id, { include: !patch.included }).catch(showError));
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = "定位水体";
      open.addEventListener("click", () => {
        setTrainingView("samples");
        state.activeRegion = patch.region || state.region;
        selectLake(patch.lake_id).catch(showError);
      });
      actions.append(include, open);
      item.append(actions);
      grid.append(item);
    }
  }

  function openPatchModal(patch) {
    state.activePatch = patch;
    renderPatchModal();
    modal.hidden = false;
  }

  function closePatchModal() {
    modal.hidden = true;
    state.activePatch = null;
  }

  function renderPatchModal() {
    const patch = state.activePatch;
    if (!patch) return;
    const name = patch.lake_display_name || patch.lake_name || patch.lake_id || patch.sample_id;
    modalTitle.textContent = name || "Patch 预览";
    modalSubtitle.textContent = patch.patch_id || "";
    modalImage.src = patch.preview_url || "";
    modalImage.alt = name ? `${name} patch` : "patch preview";
    modalToggle.textContent = patch.included ? "排除这个 patch" : "恢复包含";
    modalMeta.replaceChildren();
    for (const [label, value] of [
      ["状态", patch.included ? "include" : "exclude"],
      ["water", formatPercentText(patch.water_ratio_valid)],
      ["valid", formatPercentText(patch.valid_ratio)],
      ["ignore", patch.ignore_pixels || 0],
      ["sample", patch.sample_id || ""],
      ["image", patch.product_name || patch.image_path || ""],
    ]) {
      const row = document.createElement("div");
      row.className = "patch-modal-meta-row";
      row.innerHTML = `<span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>`;
      modalMeta.append(row);
    }
  }

  waterFilter.addEventListener("change", () => {
    state.patchWaterFilter = waterFilter.value;
    state.patchPage = 1;
    renderPatchReview();
  });
  previous.addEventListener("click", () => {
    state.patchPage -= 1;
    normalizePatchPage();
    renderPatchReview();
  });
  next.addEventListener("click", () => {
    state.patchPage += 1;
    normalizePatchPage();
    renderPatchReview();
  });
  modalClose.addEventListener("click", closePatchModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) closePatchModal();
  });
  modalToggle.addEventListener("click", () => {
    if (!state.activePatch) return;
    updateTrainingPatch(state.activePatch.patch_id, { include: !state.activePatch.included }).catch(showError);
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !modal.hidden) closePatchModal();
  });

  return { renderPatchReview, normalizePatchPage, renderPatchModal };
}
