import { useEffect, useMemo, useState } from "react";
import { Alert, Box, Button, Chip, CircularProgress, FormControl, FormControlLabel, InputLabel, LinearProgress, MenuItem, Select, Switch, TextField, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { Play, RefreshCw, Square } from "lucide-react";
import { isTrainingRunActive, type TrainingDataset, type TrainingRun, useCancelTrainingRun, useGlobalDataset, useStartTrainingRun, useTrainingDatasetConfigs, useTrainingRuns } from "./api";
import { useTrainingPatches } from "../patches/api";
import type { RegionSummary } from "../../api/types";

function percent(value?: number) { return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "-"; }
function metric(value?: number) { return Number.isFinite(value) ? Number(value).toFixed(4) : "-"; }
function statusLabel(status: string) { return ({ queued: "排队中", configured: "准备中", running: "训练中", cancel_requested: "正在取消", cancelled: "已取消", completed: "已完成", failed: "失败" } as Record<string, string>)[status] || status; }
function modelLabel(value: unknown) { return value === "pixel_mlp" ? "Pixel MLP" : "U-Net"; }

function DatasetSummary({ dataset, scope, regionOptions, source, sourcePatchCount = 0, workspacePatchCount = 0, globalPatchCount = 0, onScopeChange, onSourceChange }: { dataset?: TrainingDataset; scope: string; regionOptions: RegionSummary[]; source: "workspace" | "global"; sourcePatchCount?: number; workspacePatchCount?: number; globalPatchCount?: number; onScopeChange: (scope: string) => void; onSourceChange: (source: "workspace" | "global") => void }) {
  const values = [
    ["数据集 Patch", sourcePatchCount],
    ["可训练 Patch", dataset?.usable_patches ?? dataset?.included_patches ?? 0],
    ["样本 / 观测区域", `${dataset?.sample_count || 0} / ${dataset?.site_count || 0}`],
    ["输入", `${dataset?.in_channels || "-"} 波段 · ${(dataset?.patch_size || []).join(" x ") || "待准备"}`],
    ["水体像元占比", percent(dataset?.water_ratio)],
  ];
  return <Box sx={{ display: "grid", gap: 1 }}>
    {dataset?.error && <Alert severity="info">{dataset.error}</Alert>}
    <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", border: 1, borderColor: "divider", bgcolor: "background.paper" }}>
      <Box sx={{ px: 1.5, py: 1, borderRight: 1, borderBottom: 1, borderColor: "divider", minWidth: 0 }}>
        <Typography variant="caption" color="text.secondary" display="block">训练范围</Typography>
        <Select variant="standard" size="small" fullWidth value={scope} onChange={(event) => onScopeChange(event.target.value)} aria-label="训练范围">
          <MenuItem value="all">全部区域</MenuItem>
          {regionOptions.map((item) => <MenuItem key={item.key} value={item.key}>{item.name}</MenuItem>)}
        </Select>
      </Box>
      <Box sx={{ px: 1.5, py: 1, borderRight: 1, borderBottom: 1, borderColor: "divider", minWidth: 0 }}>
        <Typography variant="caption" color="text.secondary" display="block">数据来源</Typography>
        <Select
          variant="standard"
          size="small"
          fullWidth
          value={source}
          onChange={(event) => onSourceChange(event.target.value as "workspace" | "global")}
          aria-label="数据来源"
        >
          <MenuItem value="workspace">工作区数据集 ({workspacePatchCount})</MenuItem>
          <MenuItem value="global" disabled={!globalPatchCount}>共享数据集 ({globalPatchCount})</MenuItem>
        </Select>
      </Box>
      {values.map(([label, value]) => <Box key={label} sx={{ px: 1.5, py: 1, borderRight: 1, borderBottom: 1, borderColor: "divider", minWidth: 0 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" color="text.primary" noWrap title={String(value)}>{value}</Typography></Box>)}
    </Box>
  </Box>;
}

function RunDetails({ run }: { run?: TrainingRun }) {
  if (!run) return <Box sx={{ p: 3, textAlign: "center" }}><Typography>尚无训练任务</Typography></Box>;
  const history = run.history || [];
  const latest = history.at(-1);
  const bestIou = run.result?.best_iou;
  const config = run.config || run.options || run.result?.config || {};
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

export function TrainingView({ workspaceId, scope, regionOptions = [], onScopeChange, blocked = false, defaults = {}, selectedRunId = "", onRunSelect }: { workspaceId: string; scope: string; regionOptions?: RegionSummary[]; onScopeChange: (scope: string) => void; blocked?: boolean; defaults?: Record<string, unknown>; selectedRunId?: string; onRunSelect?: (runId: string) => void }) {
  const [datasetConfigId, setDatasetConfigId] = useState("resize256_v1");
  const [datasetSource, setDatasetSource] = useState<"workspace" | "global">("workspace");
  const workspaceRuns = useTrainingRuns(workspaceId, scope, datasetConfigId, "workspace");
  const globalRuns = useTrainingRuns(workspaceId, scope, datasetConfigId, "global");
  const query = datasetSource === "global" ? globalRuns : workspaceRuns;
  const start = useStartTrainingRun(workspaceId, scope);
  const cancel = useCancelTrainingRun(workspaceId, scope);
  const datasetConfigs = useTrainingDatasetConfigs(workspaceId, scope);
  const globalDataset = useGlobalDataset(workspaceId, scope);
  const workspacePatches = useTrainingPatches(workspaceId, scope, "included");
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
  const runs = useMemo(() => [...(workspaceRuns.data?.items || []), ...(globalRuns.data?.items || [])]
    .filter((run, index, items) => items.findIndex((item) => item.job_id === run.job_id) === index)
    .sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || ""))), [globalRuns.data?.items, workspaceRuns.data?.items]);
  const activeRun = runs.find(isTrainingRunActive);
  const selected = useMemo(() => runs.find((item) => item.job_id === selectedId) || activeRun || runs[0], [activeRun, runs, selectedId]);
  useEffect(() => { if (selected && !selectedId) setSelectedId(selected.job_id); }, [selected, selectedId]);
  useEffect(() => { if (selectedRunId && runs.some((run) => run.job_id === selectedRunId)) setSelectedId(selectedRunId); }, [runs, selectedRunId]);
  useEffect(() => {
    if (!selectedRunId) return;
    const selectedRun = runs.find((run) => run.job_id === selectedRunId);
    const source = selectedRun?.config?.dataset_source || selectedRun?.options?.dataset_source;
    if (source === "workspace" || source === "global") setDatasetSource(source);
  }, [runs, selectedRunId]);
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
  const selectRun = (runId: string) => { setSelectedId(runId); onRunSelect?.(runId); };
  const submit = () => start.mutate({ run_name: runName.trim(), model_type: modelType, epochs, batch_size: batchSize, lr, base_channels: baseChannels, hidden_channels: hiddenChannels, device, no_augment: modelType === "pixel_mlp" || noAugment, dataset_config_id: datasetConfigId, dataset_source: datasetSource }, { onSuccess: (run) => selectRun(run.job_id) });
  const sourcePatchCount = datasetSource === "global" ? globalDataset.data?.total || 0 : workspacePatches.data?.included_count || 0;

  return <Box sx={{ height: "100%", overflow: "auto", p: { xs: 1.5, sm: 2 }, display: "grid", gap: 2, alignContent: "start" }}>
    <Box component="section" sx={{ display: "grid", gap: 1 }}>
      <Box><Typography variant="h6" color="text.primary">新建训练实验</Typography><Typography variant="body2" color="text.secondary">基于当前区域和数据集创建一个可复现的训练任务</Typography></Box>
      <DatasetSummary
        dataset={query.data?.dataset}
        scope={scope}
        regionOptions={regionOptions}
        source={datasetSource}
        sourcePatchCount={sourcePatchCount}
        workspacePatchCount={workspacePatches.data?.included_count || 0}
        globalPatchCount={globalDataset.data?.total || 0}
        onScopeChange={onScopeChange}
        onSourceChange={(value) => { setDatasetSource(value); setSelectedId(""); }}
      />
    </Box>
    <Box component="section" sx={{ borderTop: 1, borderColor: "divider", pt: 2, display: "grid", gap: 1.5 }}>
      <Typography variant="subtitle1" color="text.primary">模型与训练设置</Typography>
      <Box sx={{ display: "grid", gap: 1.75 }}>
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2,minmax(180px,1fr))", lg: "minmax(220px,1.2fr) minmax(180px,1fr) minmax(140px,.7fr) auto" }, gap: 1, alignItems: "center" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: .75, minWidth: 0, minHeight: 40 }}><Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "nowrap" }}>模型架构</Typography><ToggleButtonGroup exclusive size="small" fullWidth value={modelType} onChange={(_, value: "unet" | "pixel_mlp" | null) => { if (value) setModelType(value); }} aria-label="模型类型"><ToggleButton value="unet">U-Net</ToggleButton><ToggleButton value="pixel_mlp">Pixel MLP</ToggleButton></ToggleButtonGroup></Box>
          <TextField size="small" label="实验名称" placeholder="自动生成" value={runName} onChange={(event) => setRunName(event.target.value)} />
          {modelType === "unet" && <TextField size="small" label="模型宽度" type="number" value={baseChannels} onChange={(event) => setBaseChannels(Number(event.target.value))} slotProps={{ htmlInput: { min: 4, max: 128, step: 4 } }} />}
          {modelType === "pixel_mlp" && <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}><TextField size="small" label="隐藏层 1" type="number" value={hiddenChannels[0]} onChange={(event) => setHiddenChannels([Number(event.target.value), hiddenChannels[1]])} slotProps={{ htmlInput: { min: 1, max: 1024, step: 1 } }} /><TextField size="small" label="隐藏层 2" type="number" value={hiddenChannels[1]} onChange={(event) => setHiddenChannels([hiddenChannels[0], Number(event.target.value)])} slotProps={{ htmlInput: { min: 1, max: 1024, step: 1 } }} /></Box>}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: .5, flexWrap: "wrap" }}>
            <Button size="small" variant="contained" startIcon={<Play size={15} />} disabled={blocked || Boolean(activeRun) || start.isPending || (datasetSource === "global" && !globalDataset.data?.total)} onClick={submit}>开始训练</Button>
            <Button size="small" color="error" startIcon={<Square size={14} />} disabled={!activeRun || cancel.isPending} onClick={() => activeRun && cancel.mutate(activeRun.job_id)}>取消</Button>
            <Button size="small" startIcon={<RefreshCw size={14} />} onClick={() => query.refetch()}>刷新</Button>
          </Box>
        </Box>
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2,minmax(140px,1fr))", lg: "minmax(220px,1.4fr) repeat(5,minmax(105px,1fr))" }, gap: 1, alignItems: "center" }}>
          <FormControl><InputLabel id="training-dataset-config-label">训练数据配置</InputLabel><Select labelId="training-dataset-config-label" label="训练数据配置" value={datasetConfigId} onChange={(event) => setDatasetConfigId(event.target.value)}>{(datasetConfigs.data?.items || []).map((item) => <MenuItem key={item.config_id} value={item.config_id}>{item.config.label} · {item.config.output_size} px</MenuItem>)}</Select></FormControl>
          <TextField label="轮数" type="number" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} slotProps={{ htmlInput: { min: 1, max: 500 } }} />
          <TextField label="Batch size" type="number" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} slotProps={{ htmlInput: { min: 1, max: 128 } }} />
          <TextField label="学习率" type="number" value={lr} onChange={(event) => setLr(Number(event.target.value))} slotProps={{ htmlInput: { min: .000001, max: 1, step: .0001 } }} />
          <FormControl><InputLabel id="training-device-label">运行设备</InputLabel><Select labelId="training-device-label" label="运行设备" value={device} onChange={(event) => setDevice(event.target.value)}><MenuItem value="cuda">GPU (CUDA)</MenuItem><MenuItem value="auto">自动选择</MenuItem><MenuItem value="cpu">CPU</MenuItem></Select></FormControl>
          {modelType === "unet" && <FormControlLabel sx={{ width: "fit-content", ml: 0 }} control={<Switch size="small" checked={noAugment} onChange={(event) => setNoAugment(event.target.checked)} />} label="关闭增强" />}
        </Box>
      </Box>
      {activeRun && <Typography variant="caption" color="text.secondary">已有实验正在运行</Typography>}
    </Box>
    {(start.isError || cancel.isError) && <Alert severity="error">{start.error?.message || cancel.error?.message}</Alert>}
    <Box sx={{ borderTop: 1, borderColor: "divider", minHeight: 320 }}>
      <Box sx={{ p: { xs: 1.5, sm: 2 }, minWidth: 0 }}><Typography variant="overline" color="text.secondary">当前实验详情</Typography><RunDetails run={selected} /></Box>
    </Box>
  </Box>;
}
