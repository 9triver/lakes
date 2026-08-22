import { useEffect, useMemo, useState } from "react";
import { Box, Button, Checkbox, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Pagination, Select, Typography } from "@mui/material";
import { DatabaseZap, Eye, Grid2X2Plus, LocateFixed, RotateCcw, Share2, Undo2, X } from "lucide-react";
import type { TrainingPatch } from "../../api/types";
import { useBatchManagePatches, useContributePatches, useTrainingPatches, useUpdateTrainingPatch, useWithdrawPatches } from "./api";
import { PatchGenerationControls } from "./PatchGenerationControls";

type PatchFilter = "all" | "included" | "excluded" | "contributed" | "uncontributed";

function patchId(patch: TrainingPatch) { return patch.logical_patch_id || patch.patch_id; }
function patchSiteName(patch?: TrainingPatch | null) { return patch?.site_display_name || patch?.site_name || patch?.site_id || "Patch 预览"; }
function patchSources(patch?: TrainingPatch | null) { return (patch?.label_sources || patch?.label_source || "未记录").split(",").filter(Boolean).join(" + "); }
function percent(value?: number) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }

interface PatchReviewViewProps {
  workspaceId: string;
  scope: string;
  siteId?: string;
  sourceSampleId?: string;
  onClearSource?: () => void;
  onLocate: (patch: TrainingPatch) => void;
}

export function PatchReviewView({ workspaceId, scope, siteId = "", sourceSampleId = "", onClearSource, onLocate }: PatchReviewViewProps) {
  const [filter, setFilter] = useState<PatchFilter>("all");
  const [water, setWater] = useState("");
  const [page, setPage] = useState(1);
  const [active, setActive] = useState<TrainingPatch | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [generationOpen, setGenerationOpen] = useState(false);
  const [contributionScope, setContributionScope] = useState(scope === "all" ? "all" : scope);
  const query = useTrainingPatches(workspaceId, scope, "", sourceSampleId, siteId);
  const update = useUpdateTrainingPatch(workspaceId, scope, "");
  const batch = useBatchManagePatches(workspaceId, scope);
  const contribute = useContributePatches(workspaceId, contributionScope);
  const withdraw = useWithdrawPatches(workspaceId, contributionScope);
  const allItems = query.data?.items || [];
  const selected = allItems.filter((patch) => selectedIds.has(patchId(patch)));
  const counts = useMemo(() => ({
    all: allItems.length,
    included: allItems.filter((patch) => patch.included).length,
    excluded: allItems.filter((patch) => !patch.included).length,
    contributed: allItems.filter((patch) => patch.contributed).length,
    uncontributed: allItems.filter((patch) => !patch.contributed).length,
  }), [allItems]);
  const filtered = useMemo(() => allItems.filter((patch) => {
    const statusMatches = filter === "all" || (filter === "included" && patch.included) || (filter === "excluded" && !patch.included) || (filter === "contributed" && patch.contributed) || (filter === "uncontributed" && !patch.contributed);
    const waterMatches = water === "water" ? Number(patch.water_pixels || 0) > 0 : water === "negative" ? Number(patch.water_pixels || 0) <= 0 : true;
    return statusMatches && waterMatches;
  }), [allItems, filter, water]);
  const pageSize = 18;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const items = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  const busy = update.isPending || batch.isPending || contribute.isPending || withdraw.isPending;

  useEffect(() => { setPage(1); setSelectedIds(new Set()); }, [scope, siteId, sourceSampleId, filter, water]);
  useEffect(() => { setContributionScope(scope === "all" ? "all" : scope); }, [scope]);

  const toggleSelected = (id: string) => setSelectedIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const runBatch = (operation: "include" | "exclude") => batch.mutate({ patches: selected, operation }, {
    onSuccess: () => setSelectedIds(new Set()),
    onError: (error) => { if ((error as { status?: number }).status === 409 && confirm("Workspace 中存在空间重叠 Patch，是否替换？")) batch.mutate({ patches: selected, operation, replace: true }, { onSuccess: () => setSelectedIds(new Set()) }); },
  });
  const contributeSelected = () => contribute.mutate({ patches: selected }, {
    onSuccess: () => setSelectedIds(new Set()),
    onError: (error) => { if ((error as { status?: number }).status === 409 && confirm("共享 Dataset 中存在重复或空间重叠 Patch，是否替换？")) contribute.mutate({ patches: selected, replace: true }, { onSuccess: () => setSelectedIds(new Set()) }); },
  });
  const withdrawSelected = () => withdraw.mutate(selected, { onSuccess: () => setSelectedIds(new Set()) });

  return <Box sx={{ height: "100%", overflow: "auto" }}>
    <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "grid", gap: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <Box sx={{ minWidth: 0, mr: "auto" }}><Typography variant="h6" color="text.primary">训练数据</Typography><Typography variant="caption" color="text.secondary">Workspace Patch 审核与共享 Dataset 管理</Typography></Box>
        {sourceSampleId && <Chip color="primary" variant="outlined" label={`本次生成 · ${counts.all} 个`} onDelete={onClearSource} />}
        <Button variant="outlined" startIcon={<Grid2X2Plus size={16} />} onClick={() => setGenerationOpen(true)}>重新切分</Button>
      </Box>
      <Box sx={{ display: "flex", gap: .75, flexWrap: "wrap", alignItems: "center" }}>
        {(["all", "included", "excluded", "contributed", "uncontributed"] as PatchFilter[]).map((key) => <Chip key={key} clickable color={filter === key ? "primary" : "default"} variant={filter === key ? "filled" : "outlined"} label={`${({ all: "全部", included: "已纳入", excluded: "已排除", contributed: "已贡献", uncontributed: "未贡献" } as Record<PatchFilter, string>)[key]} ${counts[key]}`} onClick={() => setFilter(key)} />)}
        <FormControl size="small" sx={{ minWidth: 130, ml: { sm: "auto" } }}><InputLabel>水体像元</InputLabel><Select label="水体像元" value={water} onChange={(event) => setWater(event.target.value)}><MenuItem value="">全部</MenuItem><MenuItem value="water">有水体</MenuItem><MenuItem value="negative">无水体</MenuItem></Select></FormControl>
      </Box>
      {selected.length > 0 && <Box sx={{ display: "flex", alignItems: "center", gap: .75, flexWrap: "wrap", pt: .75, borderTop: 1, borderColor: "divider" }}>
        <Typography variant="body2" color="text.primary">已选 {selected.length} 个</Typography>
        <Button size="small" startIcon={<RotateCcw size={15} />} disabled={busy} onClick={() => runBatch("include")}>纳入</Button>
        <Button size="small" startIcon={<X size={15} />} disabled={busy} onClick={() => runBatch("exclude")}>排除</Button>
        <FormControl size="small" sx={{ minWidth: 150 }}><InputLabel>共享范围</InputLabel><Select label="共享范围" value={contributionScope} onChange={(event) => setContributionScope(event.target.value)}>{scope !== "all" && <MenuItem value={scope}>当前区域</MenuItem>}<MenuItem value="all">全部区域</MenuItem></Select></FormControl>
        <Button size="small" startIcon={<Share2 size={15} />} disabled={busy || selected.some((patch) => !patch.included)} onClick={contributeSelected}>贡献</Button>
        <Button size="small" startIcon={<Undo2 size={15} />} disabled={busy || !selected.some((patch) => patch.contribution_scopes?.includes(contributionScope))} onClick={withdrawSelected}>撤回贡献</Button>
        <Button size="small" sx={{ ml: "auto" }} onClick={() => setSelectedIds(new Set())}>取消选择</Button>
      </Box>}
      {!selected.length && items.length > 0 && <Box sx={{ display: "flex", pt: .25 }}><Button size="small" onClick={() => setSelectedIds(new Set(items.map(patchId)))}>选择本页 ({items.length})</Button></Box>}
    </Box>
    {query.isLoading ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 240 }}><CircularProgress size={28} /></Box> : query.isError ? <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography> : items.length ? <>
      <Box sx={{ p: 1.5, display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(210px,1fr))", gap: 1 }}>
        {items.map((patch) => { const id = patchId(patch); const checked = selectedIds.has(id); return <Box key={`${patch.region || scope}:${id}`} sx={{ border: 1, borderColor: checked ? "primary.main" : patch.included ? "divider" : "error.light", bgcolor: "background.paper", opacity: patch.included ? 1 : .68 }}>
          <Box sx={{ position: "relative" }}><Box component="button" onClick={() => setActive(patch)} sx={{ border: 0, p: 0, width: "100%", aspectRatio: "1", bgcolor: "#111", cursor: "pointer" }}>{patch.preview_url && <Box component="img" src={patch.preview_url} alt="" loading="lazy" sx={{ width: "100%", height: "100%", objectFit: "contain" }} />}</Box><Checkbox checked={checked} onChange={() => toggleSelected(id)} inputProps={{ "aria-label": `选择 ${id}` }} sx={{ position: "absolute", top: 4, left: 4, bgcolor: "rgba(255,255,255,.88)", p: .25, "&:hover": { bgcolor: "white" } }} /></Box>
          <Box sx={{ p: 1, display: "grid", gap: .35 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: .5 }}><Typography noWrap variant="body2" sx={{ flex: 1 }}>{patchSiteName(patch)}</Typography><Chip size="small" color={patch.included ? "success" : "error"} variant="outlined" label={patch.included ? "已纳入" : "已排除"} /></Box>
            <Box sx={{ display: "flex", gap: .5, minHeight: 24 }}>{patch.contribution_scopes?.map((value) => <Chip key={value} size="small" icon={<DatabaseZap size={13} />} label={value === "all" ? "已贡献 · 全部" : `已贡献 · ${value}`} />)}</Box>
            <Typography variant="caption" color="text.secondary" noWrap title={patchSources(patch)}>标注 {patchSources(patch)}</Typography>
            <Typography variant="caption" color="text.secondary">水体 {percent(patch.water_ratio_valid)} · 有效 {percent(patch.valid_ratio)}</Typography>
            <Typography variant="caption" color="text.secondary" noWrap title={patch.product_name}>{patch.product_date || patch.product_name || patch.sample_id}</Typography>
            <Box sx={{ display: "flex", mt: .25 }}><Button disabled={busy} startIcon={patch.included ? <X size={14} /> : <RotateCcw size={14} />} onClick={() => update.mutate({ patch }, { onError: (error) => { if ((error as { status?: number }).status === 409 && confirm("Workspace 中存在空间重叠 Patch，是否替换？")) update.mutate({ patch, replace: true }); } })}>{patch.included ? "排除" : "纳入"}</Button><Button startIcon={<LocateFixed size={14} />} onClick={() => onLocate(patch)}>定位</Button></Box>
          </Box>
        </Box>; })}
      </Box>
      <Box sx={{ display: "flex", justifyContent: "center", py: 1 }}><Pagination count={pageCount} page={currentPage} onChange={(_, value) => setPage(value)} /></Box>
    </> : <Box sx={{ p: 4, textAlign: "center" }}><Typography>当前筛选条件下没有 Patch</Typography><Typography variant="caption" color="text.secondary">请从观测区域选择影像和标注并生成训练数据</Typography></Box>}
    <Dialog open={generationOpen} onClose={() => setGenerationOpen(false)} maxWidth="sm" fullWidth><DialogTitle>重新切分 Workspace Patch</DialogTitle><DialogContent sx={{ p: 0 }}><PatchGenerationControls workspaceId={workspaceId} scope={scope} /></DialogContent><DialogActions><Button onClick={() => setGenerationOpen(false)}>关闭</Button></DialogActions></Dialog>
    <Dialog open={Boolean(active)} onClose={() => setActive(null)} maxWidth="lg" fullWidth><DialogTitle>{patchSiteName(active)}</DialogTitle><DialogContent sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "minmax(0,1fr) 320px" }, gap: 2 }}>
      {active?.preview_url && <Box component="img" src={active.preview_url} alt="" sx={{ width: "100%", maxHeight: "70vh", objectFit: "contain", bgcolor: "#111" }} />}
      <Box sx={{ display: "grid", gap: 1, alignContent: "start", minWidth: 0 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}><Chip size="small" color={active?.included ? "success" : "error"} label={active?.included ? "已纳入 Workspace" : "已从 Workspace 排除"} />{active?.contribution_scopes?.map((value) => <Chip key={value} size="small" label={`已贡献 · ${value}`} />)}</Box>
        {[["标注来源", patchSources(active)], ["水体像元", `${Number(active?.water_pixels || 0).toLocaleString()} px (${percent(active?.water_ratio_valid)})`], ["有效像元比例", percent(active?.valid_ratio)], ["影像日期", active?.product_date || "-"], ["影像产品", active?.product_name || "-"], ["来源记录", active?.sample_id || "-"], ["Patch ID", active ? patchId(active) : "-"]].map(([label, value]) => <Box key={label} sx={{ borderBottom: 1, borderColor: "divider", pb: .75 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" color="text.primary" sx={{ wordBreak: "break-all" }}>{value}</Typography></Box>)}
      </Box>
    </DialogContent><DialogActions><Button startIcon={<Eye size={15} />} onClick={() => active && update.mutate({ patch: active })}>{active?.included ? "排除" : "纳入"}</Button><Button onClick={() => setActive(null)}>关闭</Button></DialogActions></Dialog>
  </Box>;
}
