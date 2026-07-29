import { useEffect, useState } from "react";
import { Alert, Box, Button, LinearProgress, Typography } from "@mui/material";
import { Grid2X2Plus } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { usePatchExportJob, useStartPatchExport } from "./api";

export function PatchGenerationControls({ profileId, scope }: { profileId: string; scope: string }) {
  const [jobId, setJobId] = useState("");
  const start = useStartPatchExport(profileId, scope);
  const job = usePatchExportJob(profileId, scope, jobId);
  const queryClient = useQueryClient();
  const status = job.data?.status;
  const running = start.isPending || status === "queued" || status === "running";

  useEffect(() => {
    if (status === "completed") {
      void Promise.all([
        queryClient.invalidateQueries({ queryKey: ["logical-patches"] }),
        queryClient.invalidateQueries({ queryKey: ["training-datasets"] }),
        queryClient.invalidateQueries({ queryKey: ["sites"] }),
      ]);
    }
  }, [queryClient, scope, status]);

  const submit = () => start.mutate({ patch_size: 512, stride: 512, preview_scale: 1, overwrite: true }, { onSuccess: (result) => setJobId(result.job_id) });

  return <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper" }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <Box sx={{ mr: "auto" }}><Typography variant="subtitle2">逻辑 Patch</Typography><Typography variant="caption" color="text.secondary">当前用户独立的 512 x 512 底稿，重建会保留审核状态</Typography></Box>
      <Button variant="contained" startIcon={<Grid2X2Plus size={16} />} disabled={running} onClick={submit}>重建当前用户 Patch</Button>
    </Box>
    {running && <LinearProgress variant={job.data?.progress ? "determinate" : "indeterminate"} value={job.data?.progress || 0} sx={{ mt: 1 }} />}
    {(job.data || start.isError || job.isError) && <Alert severity={status === "failed" || start.isError || job.isError ? "error" : status === "completed" ? "success" : "info"} sx={{ mt: 1, py: 0 }}>
      {start.error?.message || job.error?.message || job.data?.message || job.data?.status}
    </Alert>}
  </Box>;
}
