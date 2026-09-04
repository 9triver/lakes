import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Box, Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Typography } from "@mui/material";
import { ScanLine } from "lucide-react";
import { getJson, postJson } from "../../api/client";
import type { LocalLabelItem } from "../../api/types";
import type { SiteMapHandle } from "../map/SiteMap";
import { workspaceRegionApi } from "../workspaces/api";

export interface TrainingCaptureProps {
  workspaceId: string;
  region: string;
  siteId: string;
  jrcThreshold: number;
  localLabel?: LocalLabelItem;
  imagery: { assetId?: string; tile: string; product: string; localLabel?: LocalLabelItem | null };
  mapHandle: React.RefObject<SiteMapHandle | null>;
  modelValidation?: Record<string, unknown> | null;
  onComplete?: (sampleId: string) => void;
}

interface SaveResponse {
  sample: { sample_id: string; duplicate?: boolean; similar_samples?: unknown[] };
  patch_job?: { job_id?: string };
}

interface PatchJob {
  job_id?: string;
  status: string;
  message?: string;
  result?: { patches?: number };
  error_code?: string;
  conflicts?: Array<Record<string, unknown>>;
  pending_patch_ids?: string[];
}

export interface TrainingPatchConflict {
  sampleId: string;
  patchIds: string[];
  conflicts: Array<Record<string, unknown>>;
}

class PatchConflictError extends Error {
  constructor(public details: TrainingPatchConflict) {
    super("当前 Patch 与 Workspace 中已有 Patch 重复或空间重叠");
  }
}

async function waitForPatchJob(path: string, sampleId: string, onProgress: (message: string) => void): Promise<PatchJob> {
  let job = await getJson<PatchJob>(path);
  while (!["completed", "failed"].includes(job.status)) {
    onProgress(job.message || "正在生成 Patch");
    await new Promise((resolve) => setTimeout(resolve, 1500));
    job = await getJson<PatchJob>(path);
  }
  if (job.status === "failed") {
    if (job.error_code === "workspace_patch_conflict" && (job.pending_patch_ids || []).length) {
      throw new PatchConflictError({
        sampleId,
        patchIds: job.pending_patch_ids || [],
        conflicts: job.conflicts || [],
      });
    }
    throw new Error(job.message || "训练区域已保存，但 Patch 生成失败");
  }
  return job;
}

export interface TrainingCaptureController {
  message: string;
  isPending: boolean;
  isError: boolean;
  conflict: TrainingPatchConflict | null;
  record: () => void;
  cancelConflict: () => void;
  overwriteConflict: () => void;
}

export function useTrainingCapture({ workspaceId, region, siteId, jrcThreshold, localLabel, imagery, mapHandle, modelValidation = null, onComplete }: TrainingCaptureProps): TrainingCaptureController {
  const [message, setMessage] = useState("");
  const [conflict, setConflict] = useState<TrainingPatchConflict | null>(null);
  const queryClient = useQueryClient();
  const invalidateTrainingQueries = async () => Promise.all([
    queryClient.invalidateQueries({ queryKey: ["training-samples"] }),
    queryClient.invalidateQueries({ queryKey: ["logical-patches"] }),
    queryClient.invalidateQueries({ queryKey: ["training-datasets"] }),
    queryClient.invalidateQueries({ queryKey: ["sites"] }),
  ]);
  const save = useMutation({
    mutationFn: async () => {
      const captured = mapHandle.current?.captureView();
      if (!captured) throw new Error("地图尚未准备好");
      const selectedLocalLabel = localLabel || imagery.localLabel || undefined;
      const viewState = {
        region,
        site_id: siteId,
        ...captured,
        jrc_threshold: jrcThreshold,
        selected_local_label: selectedLocalLabel ? { id: selectedLocalLabel.id, name: selectedLocalLabel.name, path: selectedLocalLabel.path, date: selectedLocalLabel.date || "" } : null,
        selected_tile: imagery.tile,
        selected_product: imagery.product,
        selected_imagery_asset_id: imagery.assetId || imagery.product,
        model_prediction_excluded: true,
        model_validation: modelValidation,
      };
      const result = await postJson<SaveResponse>(workspaceRegionApi(workspaceId, region, `/sites/${encodeURIComponent(siteId)}/training-samples`), {
        label_source: "current_view",
        label_threshold: String(jrcThreshold),
        label_scope: "current_view",
        mask_policy: "current_view",
        buffer_ratio: 0.8,
        auto_patch: true,
        view_state: viewState,
      });
      if (result.patch_job?.job_id) {
        const job = await waitForPatchJob(
          workspaceRegionApi(workspaceId, region, `/training-patches/export-jobs/${encodeURIComponent(result.patch_job.job_id)}`),
          result.sample.sample_id,
          setMessage,
        );
        return { result, patches: job.result?.patches || 0 };
      }
      return { result, patches: null };
    },
    onSuccess: async ({ result, patches }) => {
      const similar = result.sample.similar_samples?.length || 0;
      const action = result.sample.duplicate ? "已更新已有训练区域" : "已记录训练区域";
      setMessage(`${action}${patches == null ? "" : `，生成 ${patches} 个 Patch`}${similar ? `，${similar} 个相似视图` : ""}`);
      await invalidateTrainingQueries();
      onComplete?.(result.sample.sample_id);
    },
    onError: (error) => {
      if (error instanceof PatchConflictError) {
        setConflict(error.details);
        setMessage("发现重复或空间重叠 Patch，请选择如何处理");
        return;
      }
      setMessage(error instanceof Error ? error.message : String(error));
    },
  });
  const retryPatch = useMutation({
    mutationFn: async () => {
      if (!conflict) throw new Error("没有待处理的 Patch 冲突");
      const job = await postJson<PatchJob>(workspaceRegionApi(workspaceId, region, "/training-patches/export-jobs"), {
        workspace_id: workspaceId,
        sample_id: conflict.sampleId,
        patch_ids: conflict.patchIds,
        patch_size: 256,
        stride: 128,
        preview_scale: 2,
        overwrite: true,
      });
      if (!job.job_id) throw new Error("无法创建 Patch 覆盖任务");
      const completed = await waitForPatchJob(
        workspaceRegionApi(workspaceId, region, `/training-patches/export-jobs/${encodeURIComponent(job.job_id)}`),
        conflict.sampleId,
        setMessage,
      );
      return { sampleId: conflict.sampleId, patches: completed.result?.patches || 0 };
    },
    onSuccess: async ({ sampleId, patches }) => {
      save.reset();
      setConflict(null);
      setMessage(`已覆盖冲突 Patch，当前 Patch 已纳入 Workspace（${patches} 个）`);
      await invalidateTrainingQueries();
      onComplete?.(sampleId);
    },
    onError: (error) => {
      if (error instanceof PatchConflictError) {
        setConflict(error.details);
        setMessage("仍存在 Patch 冲突，请重新选择处理方式");
        return;
      }
      setMessage(error instanceof Error ? error.message : String(error));
    },
  });

  return {
    message,
    isPending: save.isPending || retryPatch.isPending,
    isError: save.isError || retryPatch.isError,
    conflict,
    record: () => { setConflict(null); retryPatch.reset(); setMessage("保存中"); save.mutate(); },
    cancelConflict: () => { if (!retryPatch.isPending) { save.reset(); retryPatch.reset(); setConflict(null); setMessage("已取消覆盖，训练区域已保存"); } },
    overwriteConflict: () => { if (conflict && !retryPatch.isPending) { setMessage("正在覆盖冲突 Patch"); retryPatch.mutate(); } },
  };
}

export function TrainingCaptureButton({ controller }: { controller: TrainingCaptureController }) {
  return <>
    <Button variant="contained" size="small" title="从当前影像、标注和地图范围生成 Patch" startIcon={<ScanLine size={16} />} disabled={controller.isPending} onClick={controller.record}>{controller.isPending ? "生成中" : "生成训练数据"}</Button>
    <Dialog open={Boolean(controller.conflict)} onClose={controller.cancelConflict} fullWidth maxWidth="xs" aria-labelledby="training-patch-conflict-title">
      <DialogTitle id="training-patch-conflict-title">Patch 存在冲突</DialogTitle>
      <DialogContent>
        <DialogContentText>
          当前生成的 {controller.conflict?.patchIds.length || 0} 个 Patch 与 Workspace 中已有 Patch 重复或空间重叠。
          <br />覆盖后将以当前视图生成的 Patch 替换冲突的已有 Patch。
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={controller.cancelConflict} disabled={controller.isPending}>取消</Button>
        <Button variant="contained" onClick={controller.overwriteConflict} disabled={controller.isPending}>覆盖已有 Patch</Button>
      </DialogActions>
    </Dialog>
  </>;
}

export function TrainingCaptureStatus({ controller }: { controller: TrainingCaptureController }) {
  return <Typography variant="caption" color={controller.isError ? "error" : "text.secondary"}>{controller.message}</Typography>;
}

export function TrainingCapture(props: TrainingCaptureProps) {
  const controller = useTrainingCapture(props);
  return (
    <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <TrainingCaptureStatus controller={controller} />
      <TrainingCaptureButton controller={controller} />
    </Box>
  );
}
