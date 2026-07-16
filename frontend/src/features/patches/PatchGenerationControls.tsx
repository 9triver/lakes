import { useEffect, useState } from "react";
import { Alert, Box, Button, FormControlLabel, LinearProgress, Switch, TextField } from "@mui/material";
import { Grid2X2Plus } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { usePatchExportJob, useStartPatchExport } from "./api";

export function PatchGenerationControls({ scope }: { scope: string }) {
  const [patchSize, setPatchSize] = useState(256);
  const [stride, setStride] = useState(128);
  const [previewScale, setPreviewScale] = useState(2);
  const [overwrite, setOverwrite] = useState(false);
  const [jobId, setJobId] = useState("");
  const start = useStartPatchExport(scope);
  const job = usePatchExportJob(scope, jobId);
  const queryClient = useQueryClient();
  const status = job.data?.status;
  const running = start.isPending || status === "queued" || status === "running";

  useEffect(() => {
    if (status === "completed") {
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["training-patches", scope] }),
        queryClient.invalidateQueries({ queryKey: ["sites"] }),
      ]);
    }
  }, [queryClient, scope, status]);

  const submit = () => {
    start.mutate({ patch_size: patchSize, stride, preview_scale: previewScale, overwrite }, { onSuccess: (result) => setJobId(result.job_id) });
  };

  return <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <TextField label="Patch 尺寸" type="number" value={patchSize} onChange={(event) => setPatchSize(Number(event.target.value))} slotProps={{ htmlInput: { min: 64, max: 1024, step: 64 } }} sx={{ width: 130 }} />
      <TextField label="步长" type="number" value={stride} onChange={(event) => setStride(Number(event.target.value))} slotProps={{ htmlInput: { min: 32, max: 1024, step: 32 } }} sx={{ width: 110 }} />
      <TextField label="预览倍率" type="number" value={previewScale} onChange={(event) => setPreviewScale(Number(event.target.value))} slotProps={{ htmlInput: { min: 1, max: 4, step: 1 } }} sx={{ width: 120 }} />
      <FormControlLabel control={<Switch size="small" checked={overwrite} onChange={(event) => setOverwrite(event.target.checked)} />} label="覆盖重建" title="删除同参数输出后重新生成；关闭时保留已有 include/exclude 状态" />
      <Button variant="contained" startIcon={<Grid2X2Plus size={16} />} disabled={running || patchSize < 64 || stride < 32 || previewScale < 1} onClick={submit}>生成 Patch</Button>
    </Box>
    {running && <LinearProgress variant={job.data?.progress ? "determinate" : "indeterminate"} value={job.data?.progress || 0} sx={{ mt: 1 }} />}
    {(job.data || start.isError || job.isError) && <Alert severity={status === "failed" || start.isError || job.isError ? "error" : status === "completed" ? "success" : "info"} sx={{ mt: 1, py: 0 }}>
      {start.error?.message || job.error?.message || job.data?.message || job.data?.status}
    </Alert>}
  </Box>;
}
