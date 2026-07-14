import { useEffect, useMemo, useState } from "react";
import { Alert, Box, Button, Chip, CircularProgress, FormControl, FormControlLabel, InputLabel, LinearProgress, List, ListItemButton, ListItemText, MenuItem, Select, Switch, TextField, Typography } from "@mui/material";
import { Play, RefreshCw, Square } from "lucide-react";
import { isTrainingRunActive, type TrainingDataset, type TrainingRun, useCancelTrainingRun, useStartTrainingRun, useTrainingRuns } from "./api";

function percent(value?: number) { return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "-"; }
function metric(value?: number) { return Number.isFinite(value) ? Number(value).toFixed(4) : "-"; }
function statusLabel(status: string) { return ({ queued: "排队中", configured: "准备中", running: "训练中", cancel_requested: "正在取消", cancelled: "已取消", completed: "已完成", failed: "失败" } as Record<string, string>)[status] || status; }

function DatasetSummary({ dataset }: { dataset?: TrainingDataset }) {
  if (!dataset || dataset.error) return <Alert severity="warning">{dataset?.error || "还没有可用于训练的 Patch，请先生成 Patch。"}</Alert>;
  const values = [
    ["训练范围", dataset.scope === "all" ? "全部区域" : dataset.scope],
    ["区域", (dataset.regions || []).join(", ") || "-"],
    ["可用 Patch", dataset.usable_patches ?? dataset.included_patches ?? 0],
    ["包含 / 排除", `${dataset.included_patches || 0} / ${dataset.excluded_patches || 0}`],
    ["样本 / 区域", `${dataset.sample_count || 0} / ${dataset.site_count ?? dataset.site_count ?? 0}`],
    ["输入", `${dataset.in_channels || 0} 波段 · ${(dataset.patch_size || []).join(" x ")}`],
    ["水体像元", percent(dataset.water_ratio)],
  ];
  return <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", border: 1, borderColor: "divider", bgcolor: "background.paper" }}>
    {values.map(([label, value]) => <Box key={label} sx={{ px: 1.5, py: 1, borderRight: 1, borderBottom: 1, borderColor: "divider", minWidth: 0 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" color="text.primary" noWrap title={String(value)}>{value}</Typography></Box>)}
  </Box>;
}

function RunDetails({ run }: { run?: TrainingRun }) {
  if (!run) return <Box sx={{ p: 3, textAlign: "center" }}><Typography>尚无训练任务</Typography></Box>;
  const history = run.history || [];
  const latest = history.at(-1);
  const bestIou = run.result?.best_iou;
  const config = run.config || run.result?.config || {};
  return <Box sx={{ minWidth: 0 }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}><Typography variant="subtitle1" color="text.primary">{run.run_name || run.job_id}</Typography><Chip size="small" label={statusLabel(run.status)} color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : isTrainingRunActive(run) ? "primary" : "default"} /></Box>
    <LinearProgress variant="determinate" value={Number(run.progress || 0)} sx={{ mb: 1.5 }} />
    <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 1, mb: 1.5 }}>
      {[["Epoch", `${run.epoch || 0} / ${run.epochs || 0}`], ["Best val IoU", metric(bestIou)], ["Train IoU", metric(latest?.train?.iou)], ["Val IoU", metric(latest?.val?.iou)], ["模型宽度", String(config.base_channels || "-")], ["设备", String(config.device || "-")]].map(([label, value]) => <Box key={label} sx={{ borderLeft: 3, borderColor: "primary.main", pl: 1 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography color="text.primary">{value}</Typography></Box>)}
    </Box>
    <Typography variant="caption" color="text.secondary">{run.message}</Typography>
    <Box component="pre" sx={{ mt: 1.5, mb: 0, p: 1.5, bgcolor: "#171b19", color: "#dbe6df", overflow: "auto", maxHeight: 280, fontSize: 12 }}>
      {history.length ? history.slice(-12).map((item) => `epoch ${item.epoch}: train_iou=${metric(item.train?.iou)} val_iou=${metric(item.val?.iou)} train_loss=${metric(item.train?.loss)} val_loss=${metric(item.val?.loss)}`).join("\n") : "等待训练日志"}
    </Box>
  </Box>;
}

export function TrainingView({ scope }: { scope: string }) {
  const query = useTrainingRuns(scope);
  const start = useStartTrainingRun(scope);
  const cancel = useCancelTrainingRun(scope);
  const [selectedId, setSelectedId] = useState("");
  const [runName, setRunName] = useState("");
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(8);
  const [lr, setLr] = useState(0.001);
  const [baseChannels, setBaseChannels] = useState(32);
  const [device, setDevice] = useState("cuda");
  const [noAugment, setNoAugment] = useState(false);
  const runs = query.data?.items || [];
  const activeRun = runs.find(isTrainingRunActive);
  const selected = useMemo(() => runs.find((item) => item.job_id === selectedId) || activeRun || runs[0], [activeRun, runs, selectedId]);
  useEffect(() => { if (selected && !selectedId) setSelectedId(selected.job_id); }, [selected, selectedId]);

  if (query.isLoading) return <Box sx={{ display: "grid", placeItems: "center", height: "100%" }}><CircularProgress size={28} /></Box>;
  if (query.isError) return <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography>;
  const submit = () => start.mutate({ run_name: runName.trim(), epochs, batch_size: batchSize, lr, base_channels: baseChannels, device, no_augment: noAugment }, { onSuccess: (run) => setSelectedId(run.job_id) });

  return <Box sx={{ height: "100%", overflow: "auto", p: { xs: 1.5, sm: 2 }, display: "grid", gap: 2, alignContent: "start" }}>
    <DatasetSummary dataset={query.data?.dataset} />
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <TextField label="模型名称" placeholder="自动生成" value={runName} onChange={(event) => setRunName(event.target.value)} sx={{ width: 180 }} />
      <TextField label="轮数" type="number" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} sx={{ width: 95 }} slotProps={{ htmlInput: { min: 1, max: 500 } }} />
      <TextField label="Batch" type="number" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} sx={{ width: 95 }} slotProps={{ htmlInput: { min: 1, max: 128 } }} />
      <TextField label="学习率" type="number" value={lr} onChange={(event) => setLr(Number(event.target.value))} sx={{ width: 115 }} slotProps={{ htmlInput: { min: .000001, max: 1, step: .0001 } }} />
      <TextField label="模型宽度" type="number" value={baseChannels} onChange={(event) => setBaseChannels(Number(event.target.value))} sx={{ width: 115 }} slotProps={{ htmlInput: { min: 4, max: 128, step: 4 } }} />
      <FormControl sx={{ width: 105 }}><InputLabel>设备</InputLabel><Select label="设备" value={device} onChange={(event) => setDevice(event.target.value)}><MenuItem value="cuda">GPU</MenuItem><MenuItem value="auto">自动</MenuItem><MenuItem value="cpu">CPU</MenuItem></Select></FormControl>
      <FormControlLabel control={<Switch size="small" checked={noAugment} onChange={(event) => setNoAugment(event.target.checked)} />} label="关闭增强" />
      <Button variant="contained" startIcon={<Play size={16} />} disabled={Boolean(activeRun) || start.isPending || !query.data?.dataset?.usable_patches} onClick={submit}>开始训练</Button>
      <Button color="error" startIcon={<Square size={15} />} disabled={!activeRun || cancel.isPending} onClick={() => activeRun && cancel.mutate(activeRun.job_id)}>取消</Button>
      <Button startIcon={<RefreshCw size={15} />} onClick={() => query.refetch()}>刷新</Button>
    </Box>
    {(start.isError || cancel.isError) && <Alert severity="error">{start.error?.message || cancel.error?.message}</Alert>}
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "300px minmax(0,1fr)" }, borderTop: 1, borderColor: "divider", minHeight: 320 }}>
      <Box sx={{ borderRight: { lg: 1 }, borderBottom: { xs: 1, lg: 0 }, borderColor: "divider", maxHeight: 430, overflow: "auto" }}><Typography variant="overline" sx={{ px: 1.5 }}>历史任务 ({runs.length})</Typography><List dense disablePadding>{runs.map((run) => <ListItemButton key={run.job_id} selected={run.job_id === selected?.job_id} onClick={() => setSelectedId(run.job_id)}><ListItemText primary={run.run_name || run.job_id} secondary={`${statusLabel(run.status)} · ${run.updated_at || run.created_at || ""}`} /></ListItemButton>)}</List></Box>
      <Box sx={{ p: { xs: 1.5, sm: 2 }, minWidth: 0 }}><RunDetails run={selected} /></Box>
    </Box>
  </Box>;
}
