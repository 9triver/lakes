import { useEffect, useMemo, useState } from "react";
import { Alert, Box, Button, Chip, CircularProgress, FormControl, FormControlLabel, InputLabel, LinearProgress, List, ListItemButton, ListItemText, MenuItem, Select, Switch, TextField, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { Play, RefreshCw, Square } from "lucide-react";
import { isTrainingRunActive, type TrainingDataset, type TrainingRun, useCancelTrainingRun, useGlobalDataset, useStartTrainingRun, useTrainingDatasetConfigs, useTrainingRuns } from "./api";

function percent(value?: number) { return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "-"; }
function metric(value?: number) { return Number.isFinite(value) ? Number(value).toFixed(4) : "-"; }
function statusLabel(status: string) { return ({ queued: "排队中", configured: "准备中", running: "训练中", cancel_requested: "正在取消", cancelled: "已取消", completed: "已完成", failed: "失败" } as Record<string, string>)[status] || status; }
function modelLabel(value: unknown) { return value === "pixel_mlp" ? "Pixel MLP" : "U-Net"; }

function DatasetSummary({ dataset, source, globalPatchCount = 0 }: { dataset?: TrainingDataset; source: "workspace" | "global"; globalPatchCount?: number }) {
  if (!dataset || dataset.error) return <Alert severity="info">{dataset?.error || "开始训练时将自动准备数据。"}</Alert>;
  const values = [
    ["数据来源", source === "global" ? "共享 Dataset" : "当前 Workspace"],
    ["训练范围", dataset.scope === "all" ? "全部区域" : dataset.scope],
    ["区域", (dataset.regions || []).join(", ") || "-"],
    ["可用 Patch", dataset.usable_patches ?? dataset.included_patches ?? 0],
    ["包含 / 排除", `${dataset.included_patches || 0} / ${dataset.excluded_patches || 0}`],
    ["样本 / 区域", `${dataset.sample_count || 0} / ${dataset.site_count ?? dataset.site_count ?? 0}`],
    ["输入", `${dataset.in_channels || 0} 波段 · ${(dataset.patch_size || []).join(" x ")}`],
    ["水体像元", percent(dataset.water_ratio)],
    ["全局 Dataset", globalPatchCount],
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
  const architecture = String(config.architecture_label || (config.model_type === "pixel_mlp" ? "5 -> 16 -> 8 -> 1" : `U-Net (base ${config.base_channels || "-"})`));
  return <Box sx={{ minWidth: 0 }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}><Typography variant="subtitle1" color="text.primary">{run.run_name || run.job_id}</Typography><Chip size="small" label={statusLabel(run.status)} color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : isTrainingRunActive(run) ? "primary" : "default"} /></Box>
    <LinearProgress variant="determinate" value={Number(run.progress || 0)} sx={{ mb: 1.5 }} />
    <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 1, mb: 1.5 }}>
      {[["Epoch", `${run.epoch || 0} / ${run.epochs || 0}`], ["Best val IoU", metric(bestIou)], ["Train IoU", metric(latest?.train?.iou)], ["Val IoU", metric(latest?.val?.iou)], ["数据来源", config.dataset_source === "global" ? "共享 Dataset" : "当前 Workspace"], ["模型", modelLabel(config.model_type)], ["架构", architecture], ["设备", String(config.device || "-")]].map(([label, value]) => <Box key={label} sx={{ borderLeft: 3, borderColor: "primary.main", pl: 1, minWidth: 0 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography color="text.primary" noWrap title={value}>{value}</Typography></Box>)}
    </Box>
    <Typography variant="caption" color="text.secondary">{run.message}</Typography>
    <Box component="pre" sx={{ mt: 1.5, mb: 0, p: 1.5, bgcolor: "#171b19", color: "#dbe6df", overflow: "auto", maxHeight: 280, fontSize: 12 }}>
      {history.length ? history.slice(-12).map((item) => `epoch ${item.epoch}: train_iou=${metric(item.train?.iou)} val_iou=${metric(item.val?.iou)} train_loss=${metric(item.train?.loss)} val_loss=${metric(item.val?.loss)}`).join("\n") : "等待训练日志"}
    </Box>
  </Box>;
}

export function TrainingView({ workspaceId, scope, blocked = false, defaults = {} }: { workspaceId: string; scope: string; blocked?: boolean; defaults?: Record<string, unknown> }) {
  const [datasetConfigId, setDatasetConfigId] = useState("resize256_v1");
  const [datasetSource, setDatasetSource] = useState<"workspace" | "global">("workspace");
  const query = useTrainingRuns(workspaceId, scope, datasetConfigId, datasetSource);
  const start = useStartTrainingRun(workspaceId, scope);
  const cancel = useCancelTrainingRun(workspaceId, scope);
  const datasetConfigs = useTrainingDatasetConfigs(workspaceId, scope);
  const globalDataset = useGlobalDataset(workspaceId, scope);
  const [selectedId, setSelectedId] = useState("");
  const [runName, setRunName] = useState("");
  const [modelType, setModelType] = useState<"unet" | "pixel_mlp">("unet");
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(8);
  const [lr, setLr] = useState(0.001);
  const [baseChannels, setBaseChannels] = useState(32);
  const [hiddenChannels, setHiddenChannels] = useState<[number, number]>([16, 8]);
  const [device, setDevice] = useState("cuda");
  const [noAugment, setNoAugment] = useState(false);
  const runs = query.data?.items || [];
  const activeRun = runs.find(isTrainingRunActive);
  const selected = useMemo(() => runs.find((item) => item.job_id === selectedId) || activeRun || runs[0], [activeRun, runs, selectedId]);
  useEffect(() => { if (selected && !selectedId) setSelectedId(selected.job_id); }, [selected, selectedId]);
  useEffect(() => {
    setDatasetConfigId(String(defaults.dataset_config_id || "resize256_v1"));
    setDatasetSource(defaults.dataset_source === "global" ? "global" : "workspace");
    setModelType(defaults.model_type === "pixel_mlp" ? "pixel_mlp" : "unet");
    setEpochs(Number(defaults.epochs || 30));
    setBatchSize(Number(defaults.batch_size || 8));
    setLr(Number(defaults.lr || .001));
    setBaseChannels(Number(defaults.base_channels || 32));
    const hidden = Array.isArray(defaults.hidden_channels) ? defaults.hidden_channels : [16, 8];
    setHiddenChannels([Number(hidden[0] || 16), Number(hidden[1] || 8)]);
    setDevice(String(defaults.device || "cuda"));
    setNoAugment(Boolean(defaults.no_augment));
    setRunName("");
    setSelectedId("");
  }, [workspaceId]);

  if (query.isLoading) return <Box sx={{ display: "grid", placeItems: "center", height: "100%" }}><CircularProgress size={28} /></Box>;
  if (query.isError) return <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography>;
  const submit = () => start.mutate({ run_name: runName.trim(), model_type: modelType, epochs, batch_size: batchSize, lr, base_channels: baseChannels, hidden_channels: hiddenChannels, device, no_augment: modelType === "pixel_mlp" || noAugment, dataset_config_id: datasetConfigId, dataset_source: datasetSource }, { onSuccess: (run) => setSelectedId(run.job_id) });

  return <Box sx={{ height: "100%", overflow: "auto", p: { xs: 1.5, sm: 2 }, display: "grid", gap: 2, alignContent: "start" }}>
    <DatasetSummary dataset={query.data?.dataset} source={datasetSource} globalPatchCount={globalDataset.data?.total || 0} />
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <FormControl sx={{ minWidth: 190 }}><InputLabel id="training-dataset-source-label">数据来源</InputLabel><Select labelId="training-dataset-source-label" label="数据来源" value={datasetSource} onChange={(event) => { setDatasetSource(event.target.value as "workspace" | "global"); setSelectedId(""); }}><MenuItem value="workspace">当前 Workspace</MenuItem><MenuItem value="global" disabled={!globalDataset.data?.total}>共享 Dataset ({globalDataset.data?.total || 0})</MenuItem></Select></FormControl>
      <FormControl sx={{ minWidth: 210 }}><InputLabel id="training-dataset-config-label">训练数据配置</InputLabel><Select labelId="training-dataset-config-label" label="训练数据配置" value={datasetConfigId} onChange={(event) => setDatasetConfigId(event.target.value)}>{(datasetConfigs.data?.items || []).map((item) => <MenuItem key={item.config_id} value={item.config_id}>{item.config.label} · {item.config.output_size} px</MenuItem>)}</Select></FormControl>
      <ToggleButtonGroup exclusive size="small" value={modelType} onChange={(_, value: "unet" | "pixel_mlp" | null) => { if (value) setModelType(value); }} aria-label="模型类型">
        <ToggleButton value="unet">U-Net</ToggleButton>
        <ToggleButton value="pixel_mlp">Pixel MLP</ToggleButton>
      </ToggleButtonGroup>
      <TextField label="模型名称" placeholder="自动生成" value={runName} onChange={(event) => setRunName(event.target.value)} sx={{ width: 180 }} />
      <TextField label="轮数" type="number" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} sx={{ width: 95 }} slotProps={{ htmlInput: { min: 1, max: 500 } }} />
      <TextField label="Batch" type="number" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} sx={{ width: 95 }} slotProps={{ htmlInput: { min: 1, max: 128 } }} />
      <TextField label="学习率" type="number" value={lr} onChange={(event) => setLr(Number(event.target.value))} sx={{ width: 115 }} slotProps={{ htmlInput: { min: .000001, max: 1, step: .0001 } }} />
      {modelType === "unet" && <TextField label="模型宽度" type="number" value={baseChannels} onChange={(event) => setBaseChannels(Number(event.target.value))} sx={{ width: 115 }} slotProps={{ htmlInput: { min: 4, max: 128, step: 4 } }} />}
      {modelType === "pixel_mlp" && <TextField label="隐藏层 1" type="number" value={hiddenChannels[0]} onChange={(event) => setHiddenChannels([Number(event.target.value), hiddenChannels[1]])} sx={{ width: 105 }} slotProps={{ htmlInput: { min: 1, max: 1024, step: 1 } }} />}
      {modelType === "pixel_mlp" && <TextField label="隐藏层 2" type="number" value={hiddenChannels[1]} onChange={(event) => setHiddenChannels([hiddenChannels[0], Number(event.target.value)])} sx={{ width: 105 }} slotProps={{ htmlInput: { min: 1, max: 1024, step: 1 } }} />}
      <FormControl sx={{ width: 105 }}><InputLabel id="training-device-label">设备</InputLabel><Select labelId="training-device-label" label="设备" value={device} onChange={(event) => setDevice(event.target.value)}><MenuItem value="cuda">GPU</MenuItem><MenuItem value="auto">自动</MenuItem><MenuItem value="cpu">CPU</MenuItem></Select></FormControl>
      {modelType === "unet" && <FormControlLabel control={<Switch size="small" checked={noAugment} onChange={(event) => setNoAugment(event.target.checked)} />} label="关闭增强" />}
      <Button variant="contained" startIcon={<Play size={16} />} disabled={blocked || Boolean(activeRun) || start.isPending || (datasetSource === "global" && !globalDataset.data?.total)} onClick={submit}>开始训练</Button>
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
