import { useState } from "react";
import type { ReactNode } from "react";
import { Alert, Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Divider, IconButton, List, ListItem, ListItemButton, ListItemText, Typography } from "@mui/material";
import { Trash2 } from "lucide-react";
import { isTrainingRunActive, useDeleteTrainingRun, useGlobalDataset, useTrainingRuns } from "../training/api";
import { useTrainingPatches } from "../patches/api";

interface ModelExperimentSidebarProps {
  workspaceId: string;
  scope: string;
  regionControl: ReactNode;
  showRegionControl?: boolean;
  selectedRunId?: string;
  onSelectRun: (runId: string) => void;
}

function statusLabel(status: string) {
  return ({ queued: "排队中", configured: "准备中", running: "训练中", cancel_requested: "正在取消", cancelled: "已取消", completed: "已完成", failed: "失败" } as Record<string, string>)[status] || status;
}

function statusColor(status: string): "default" | "primary" | "success" | "error" {
  if (status === "completed") return "success";
  if (status === "failed") return "error";
  if (["queued", "configured", "running", "cancel_requested"].includes(status)) return "primary";
  return "default";
}

function metric(value?: number) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "-";
}

export function ModelExperimentSidebar({ workspaceId, scope, regionControl, showRegionControl = true, selectedRunId = "", onSelectRun }: ModelExperimentSidebarProps) {
  const workspaceRuns = useTrainingRuns(workspaceId, scope, "resize256_v1", "workspace");
  const globalRuns = useTrainingRuns(workspaceId, scope, "resize256_v1", "global");
  const workspacePatches = useTrainingPatches(workspaceId, scope, "included");
  const globalDataset = useGlobalDataset(workspaceId, scope);
  const deleteRun = useDeleteTrainingRun(workspaceId, scope);
  const [pendingDeleteId, setPendingDeleteId] = useState("");
  const runs = [...(workspaceRuns.data?.items || []), ...(globalRuns.data?.items || [])]
    .filter((run, index, items) => items.findIndex((item) => item.job_id === run.job_id) === index)
    .sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")));
  const workspaceItems = workspacePatches.data?.items || [];
  const workspaceCount = workspacePatches.data?.included_count || 0;
  const globalCount = globalDataset.data?.total || 0;
  const workspaceSites = new Set(workspaceItems.map((patch) => `${patch.region || scope}:${patch.site_id}`)).size;
  const workspaceSamples = new Set(workspaceItems.map((patch) => patch.sample_id)).size;
  const pendingDeleteRun = runs.find((run) => run.job_id === pendingDeleteId);
  const closeDeleteDialog = () => {
    if (deleteRun.isPending) return;
    setPendingDeleteId("");
    deleteRun.reset();
  };
  const confirmDelete = () => {
    if (!pendingDeleteRun) return;
    deleteRun.mutate(pendingDeleteRun.job_id, {
      onSuccess: () => {
        if (pendingDeleteRun.job_id === selectedRunId) onSelectRun("");
        setPendingDeleteId("");
      },
    });
  };

  return <Box sx={{ minHeight: 0, height: "100%", overflow: "hidden", display: "grid", gridTemplateRows: "auto minmax(0,1fr)" }}>
    <Box sx={{ px: 2, py: 1.25, display: "grid", gap: 1, borderBottom: 1, borderColor: "divider" }}>
      {showRegionControl && <><Typography variant="overline" color="text.secondary" sx={{ lineHeight: 1.5 }}>实验范围</Typography>{regionControl}</>}
      <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: .75 }}>
        <Box sx={{ p: 1, bgcolor: "action.hover", minWidth: 0 }}><Typography variant="caption" color="text.secondary" display="block">工作区数据集</Typography><Typography variant="subtitle2" color="text.primary">{workspaceCount} 个 Patch</Typography></Box>
        <Box sx={{ p: 1, bgcolor: "action.hover", minWidth: 0 }}><Typography variant="caption" color="text.secondary" display="block">共享数据集</Typography><Typography variant="subtitle2" color="text.primary">{globalCount} 个 Patch</Typography></Box>
      </Box>
      <Typography variant="caption" color="text.secondary">{workspaceSamples} 个样本 · {workspaceSites} 个观测区域 · 已纳入 Patch</Typography>
      <Typography variant="caption" color="text.secondary">训练数据可在“训练数据”页面管理</Typography>
    </Box>
    <Box sx={{ minHeight: 0, overflow: "auto" }}>
      <Box sx={{ px: 2, pt: 1.5, pb: .75 }}>
        <Typography variant="overline" color="text.secondary">历史实验 ({runs.length})</Typography>
      </Box>
      {workspaceRuns.isLoading || globalRuns.isLoading ? <Box sx={{ display: "grid", placeItems: "center", p: 3 }}><CircularProgress size={24} /></Box>
        : !runs.length ? <Box sx={{ px: 2, py: 2, textAlign: "center" }}><Typography variant="body2" color="text.secondary">尚无训练实验</Typography><Typography variant="caption" color="text.secondary">在右侧配置参数并开始训练</Typography></Box>
          : <List disablePadding>{runs.map((run) => {
            const bestIou = run.result?.best_iou;
            const source = (run.config?.dataset_source || run.options?.dataset_source) === "global" ? "共享" : "工作区";
            const deletable = !isTrainingRunActive(run);
            return <ListItem key={run.job_id} disablePadding secondaryAction={deletable ? <IconButton edge="end" size="small" color="error" aria-label="删除实验" title="删除实验" disabled={deleteRun.isPending} onClick={(event) => { event.stopPropagation(); deleteRun.reset(); setPendingDeleteId(run.job_id); }}><Trash2 size={16} /></IconButton> : undefined}>
              <ListItemButton selected={run.job_id === selectedRunId} onClick={() => onSelectRun(run.job_id)} sx={{ alignItems: "flex-start", py: 1, pr: deletable ? 6 : 2 }}>
                <ListItemText
                  primary={<Box sx={{ display: "flex", alignItems: "center", gap: .5, minWidth: 0 }}><Typography variant="body2" noWrap sx={{ minWidth: 0, flex: 1 }}>{run.run_name || run.job_id}</Typography><Chip size="small" color={statusColor(run.status)} label={statusLabel(run.status)} /></Box>}
                  secondary={`${source} · ${run.scope === "all" ? "全部区域" : run.scope} · IoU ${metric(bestIou)}${run.updated_at ? ` · ${run.updated_at}` : ""}`}
                />
              </ListItemButton>
            </ListItem>;
          })}</List>}
      {(workspaceRuns.isError || globalRuns.isError || deleteRun.isError) && <><Divider /><Typography color="error" variant="caption" sx={{ display: "block", p: 1.5 }}>{deleteRun.error?.message || workspaceRuns.error?.message || globalRuns.error?.message}</Typography></>}
    </Box>
    <Dialog open={Boolean(pendingDeleteRun)} onClose={closeDeleteDialog} fullWidth maxWidth="xs" aria-labelledby="delete-training-run-title">
      <DialogTitle id="delete-training-run-title">删除训练实验</DialogTitle>
      <DialogContent>
        <DialogContentText>
          确定删除“{pendingDeleteRun?.run_name || pendingDeleteRun?.job_id}”吗？
          <br />模型权重和训练日志将被删除，训练数据集不会受到影响。
        </DialogContentText>
        {deleteRun.isError && <Alert severity="error" sx={{ mt: 2 }}>{deleteRun.error.message}</Alert>}
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={closeDeleteDialog} disabled={deleteRun.isPending}>取消</Button>
        <Button color="error" variant="contained" onClick={confirmDelete} disabled={deleteRun.isPending}>
          {deleteRun.isPending ? "删除中..." : "确认删除"}
        </Button>
      </DialogActions>
    </Dialog>
  </Box>;
}
