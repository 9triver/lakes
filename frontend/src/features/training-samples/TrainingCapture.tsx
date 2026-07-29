import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Box, Button, TextField, Typography } from "@mui/material";
import { ScanLine } from "lucide-react";
import { getJson, postJson } from "../../api/client";
import type { LocalLabelItem } from "../../api/types";
import type { SiteMapHandle } from "../map/SiteMap";
import { profileRegionApi } from "../profiles/api";

interface TrainingCaptureProps {
  profileId: string;
  region: string;
  siteId: string;
  jrcThreshold: number;
  localLabel?: LocalLabelItem;
  imagery: { tile: string; product: string };
  mapHandle: React.RefObject<SiteMapHandle | null>;
  modelValidation?: Record<string, unknown> | null;
}

interface SaveResponse {
  sample: { sample_id: string; duplicate?: boolean; similar_samples?: unknown[] };
  patch_job?: { job_id?: string };
}

export function TrainingCapture({ profileId, region, siteId, jrcThreshold, localLabel, imagery, mapHandle, modelValidation = null }: TrainingCaptureProps) {
  const [notes, setNotes] = useState("");
  const [message, setMessage] = useState("");
  const queryClient = useQueryClient();
  const save = useMutation({
    mutationFn: async () => {
      const captured = mapHandle.current?.captureView();
      if (!captured) throw new Error("地图尚未准备好");
      const viewState = {
        region,
        site_id: siteId,
        ...captured,
        jrc_threshold: jrcThreshold,
        selected_local_label: localLabel ? { id: localLabel.id, name: localLabel.name, path: localLabel.path, date: localLabel.date || "" } : null,
        selected_tile: imagery.tile,
        selected_product: imagery.product,
        model_prediction_excluded: true,
        model_validation: modelValidation,
      };
      const result = await postJson<SaveResponse>(profileRegionApi(profileId, region, `/sites/${encodeURIComponent(siteId)}/training-samples`), {
        label_source: "current_view",
        label_threshold: String(jrcThreshold),
        label_scope: "current_view",
        mask_policy: "current_view",
        notes,
        buffer_ratio: 0.8,
        auto_patch: true,
        view_state: viewState,
      });
      if (result.patch_job?.job_id) {
        let job = await getJson<{ status: string; message?: string; result?: { patches?: number } }>(profileRegionApi(profileId, region, `/training-patches/export-jobs/${encodeURIComponent(result.patch_job.job_id)}`));
        while (!["completed", "failed"].includes(job.status)) {
          setMessage(job.message || "正在生成逻辑 Patch");
          await new Promise((resolve) => setTimeout(resolve, 1500));
          job = await getJson(profileRegionApi(profileId, region, `/training-patches/export-jobs/${encodeURIComponent(result.patch_job!.job_id!)}`));
        }
        if (job.status === "failed") throw new Error(job.message || "训练区域已保存，但逻辑 Patch 生成失败");
        return { result, patches: job.result?.patches || 0 };
      }
      return { result, patches: null };
    },
    onSuccess: async ({ result, patches }) => {
      const similar = result.sample.similar_samples?.length || 0;
      const action = result.sample.duplicate ? "已更新已有训练区域" : "已记录训练区域";
      setMessage(`${action}${patches == null ? "" : `，生成 ${patches} 个逻辑 Patch`}${similar ? `，${similar} 个相似视图` : ""}`);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["training-samples"] }),
        queryClient.invalidateQueries({ queryKey: ["logical-patches"] }),
        queryClient.invalidateQueries({ queryKey: ["training-datasets"] }),
        queryClient.invalidateQueries({ queryKey: ["sites"] }),
      ]);
    },
    onError: (error) => setMessage(error.message),
  });

  return (
    <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <TextField value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="备注" sx={{ minWidth: 220, flex: 1 }} />
      <Button variant="contained" startIcon={<ScanLine size={16} />} disabled={save.isPending} onClick={() => { setMessage("保存中"); save.mutate(); }}>{save.isPending ? "处理中" : "记录当前视图为训练区域"}</Button>
      <Typography variant="caption" color={save.isError ? "error" : "text.secondary"}>{message}</Typography>
    </Box>
  );
}
