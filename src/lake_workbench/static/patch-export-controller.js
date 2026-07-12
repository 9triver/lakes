import { fetchJson, postJson } from "./api.js";

export function createPatchExportController({ elements, apiPath, loadTrainingPatches, showError }) {
  const { patchSize, stride, previewScale, overwrite, start, status } = elements;

  async function startPatchExport() {
    start.disabled = true;
    status.textContent = "提交生成任务";
    const payload = {
      patch_size: Number(patchSize.value || 256),
      stride: Number(stride.value || 128),
      preview_scale: Number(previewScale.value || 2),
      overwrite: overwrite.checked,
    };
    try {
      const job = await postJson(apiPath("/training-patches/export-jobs"), payload);
      pollPatchExportJob(job.job_id).catch(showError);
    } catch (error) {
      start.disabled = false;
      throw error;
    }
  }

  async function pollPatchExportJob(jobId) {
    const job = await fetchJson(apiPath(`/training-patches/export-jobs/${encodeURIComponent(jobId)}`));
    status.textContent = job.message || job.status || "处理中";
    if (job.status === "completed") {
      start.disabled = false;
      await loadTrainingPatches();
      return;
    }
    if (job.status === "failed") {
      start.disabled = false;
      throw new Error(job.message || "Patch 生成失败");
    }
    setTimeout(() => pollPatchExportJob(jobId).catch(showError), 1500);
  }

  start.addEventListener("click", () => startPatchExport().catch(showError));
  return { startPatchExport, pollPatchExportJob };
}
