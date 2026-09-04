import { useEffect, useMemo, useState } from "react";
import { Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, FormControlLabel, InputLabel, MenuItem, Pagination, Select, Switch, Typography } from "@mui/material";
import { RotateCcw, Share2, X } from "lucide-react";
import type { TrainingPatch } from "../../api/types";
import { useContributePatches, useTrainingPatches, useUpdateTrainingPatch } from "./api";

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
}

export function PatchReviewView({ workspaceId, scope, siteId = "", sourceSampleId = "", onClearSource }: PatchReviewViewProps) {
  const [filter, setFilter] = useState<PatchFilter>("all");
  const [water, setWater] = useState("");
  const [page, setPage] = useState(1);
  const [active, setActive] = useState<TrainingPatch | null>(null);
  const [showLabelOverlay, setShowLabelOverlay] = useState(true);
  const query = useTrainingPatches(workspaceId, scope, "", sourceSampleId, siteId);
  const update = useUpdateTrainingPatch(workspaceId, scope, "");
  const contribute = useContributePatches(workspaceId, scope);
  const allItems = query.data?.items || [];
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
  const busy = update.isPending || contribute.isPending;
  const activeShared = Boolean(active?.contribution_scopes?.includes(scope));
  const activePreviewUrl = active?.preview_url ? `${active.preview_url}${active.preview_url.includes("?") ? "&" : "?"}overlay=${showLabelOverlay ? "1" : "0"}` : "";

  useEffect(() => { setPage(1); }, [scope, siteId, sourceSampleId, filter, water]);
  const contributePatch = (patch: TrainingPatch) => contribute.mutate({ patches: [patch] }, {
    onSuccess: () => {
      if (active && patchId(active) === patchId(patch)) setActive({ ...active, contributed: true, contribution_scopes: [...new Set([...(active.contribution_scopes || []), scope])] });
    },
    onError: (error) => { if ((error as { status?: number }).status === 409 && confirm("目标 Dataset 中存在重复或空间重叠 Patch，是否替换？")) contribute.mutate({ patches: [patch], replace: true }); },
  });

  return <Box sx={{ height: "100%", overflow: "auto" }}>
    <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: "divider", bgcolor: "background.paper", display: "grid", gap: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <Box sx={{ minWidth: 0, mr: "auto" }}><Typography variant="h6" color="text.primary">训练数据</Typography><Typography variant="caption" color="text.secondary">Workspace Patch 审核与共享 Dataset 管理</Typography></Box>
        {sourceSampleId && <Chip color="primary" variant="outlined" label={`本次生成 · ${counts.all} 个`} onDelete={onClearSource} />}
      </Box>
      <Box sx={{ display: "flex", gap: .75, flexWrap: "wrap", alignItems: "center" }}>
        {(["all", "included", "excluded", "contributed", "uncontributed"] as PatchFilter[]).map((key) => <Chip key={key} clickable color={filter === key ? "primary" : "default"} variant={filter === key ? "filled" : "outlined"} label={`${({ all: "全部", included: "已纳入", excluded: "已排除", contributed: "已共享", uncontributed: "未共享" } as Record<PatchFilter, string>)[key]} ${counts[key]}`} onClick={() => setFilter(key)} />)}
        <FormControl size="small" sx={{ minWidth: 130, ml: { sm: "auto" } }}><InputLabel>水体像元</InputLabel><Select label="水体像元" value={water} onChange={(event) => setWater(event.target.value)}><MenuItem value="">全部</MenuItem><MenuItem value="water">有水体</MenuItem><MenuItem value="negative">无水体</MenuItem></Select></FormControl>
      </Box>
    </Box>
    {query.isLoading ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 240 }}><CircularProgress size={28} /></Box> : query.isError ? <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography> : items.length ? <>
      <Box sx={{ p: 1.5, display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(210px,1fr))", gap: 1 }}>
        {items.map((patch) => { const id = patchId(patch); const shared = patch.contribution_scopes?.includes(scope); return <Box key={`${patch.region || scope}:${id}`} sx={{ border: 1, borderColor: patch.included ? "divider" : "error.light", bgcolor: "background.paper", opacity: patch.included ? 1 : .68, overflow: "hidden" }}>
          <Box sx={{ position: "relative" }}>
            <Box component="button" onClick={() => { setActive(patch); setShowLabelOverlay(true); }} aria-label={`查看 ${patchSiteName(patch)} Patch`} sx={{ border: 0, p: 0, display: "block", width: "100%", aspectRatio: "1", bgcolor: "#111", cursor: "pointer" }}>{patch.preview_url && <Box component="img" src={patch.preview_url} alt="" loading="lazy" sx={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }} />}</Box>
            <Chip size="small" color={patch.included ? "success" : "error"} variant="filled" label={patch.included ? "已纳入" : "已排除"} sx={{ position: "absolute", top: .75, right: .75, fontWeight: 600 }} />
          </Box>
          <Box sx={{ p: 1.25, display: "grid", gap: .5, minWidth: 0 }}>
            <Typography noWrap variant="body2" color="text.primary" sx={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }} title={patchSiteName(patch)}>{patchSiteName(patch)}</Typography>
            <Typography variant="caption" color="text.secondary" noWrap title={patch.product_name}>{patch.product_date || patch.product_name || patch.sample_id}</Typography>
            <Box sx={{ display: "flex", gap: 1.25, flexWrap: "wrap" }}>
              <Typography variant="caption" color="text.secondary">水体 {percent(patch.water_ratio_valid)}</Typography>
              <Typography variant="caption" color="text.secondary">有效 {percent(patch.valid_ratio)}</Typography>
            </Box>
            <Box sx={{ display: "flex", mt: .25, gap: .25 }}>
              <Button fullWidth disabled={busy} startIcon={patch.included ? <X size={14} /> : <RotateCcw size={14} />} onClick={() => update.mutate({ patch }, { onError: (error) => { if ((error as { status?: number }).status === 409 && confirm("Workspace 中存在空间重叠 Patch，是否替换？")) update.mutate({ patch, replace: true }); } })}>{patch.included ? "排除" : "纳入"}</Button>
              <Button fullWidth disabled={busy || !patch.included || shared} startIcon={<Share2 size={14} />} onClick={() => contributePatch(patch)}>{shared ? "已共享" : "共享"}</Button>
            </Box>
          </Box>
        </Box>; })}
      </Box>
      <Box sx={{ display: "flex", justifyContent: "center", py: 1 }}><Pagination count={pageCount} page={currentPage} onChange={(_, value) => setPage(value)} /></Box>
    </> : <Box sx={{ p: 4, textAlign: "center" }}><Typography>当前筛选条件下没有 Patch</Typography><Typography variant="caption" color="text.secondary">请从观测区域选择影像和标注并生成训练数据</Typography></Box>}
    <Dialog open={Boolean(active)} onClose={() => setActive(null)} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ pb: 1.25, borderBottom: 1, borderColor: "divider" }}>
        <Box sx={{ display: "flex", alignItems: "flex-start", gap: 2, flexWrap: "wrap" }}>
          <Box sx={{ minWidth: 0, flex: "1 1 280px" }}>
            <Typography variant="h6" color="text.primary" noWrap title={patchSiteName(active)}>{patchSiteName(active)}</Typography>
            <Typography variant="body2" color="text.secondary" noWrap title={active?.product_name || ""}>{active?.product_date || "未记录日期"}{active?.product_name ? ` · ${active.product_name}` : ""}</Typography>
          </Box>
          <Box sx={{ display: "flex", gap: .75, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <Chip size="small" color={active?.included ? "success" : "error"} label={active?.included ? "已纳入数据集" : "已从数据集排除"} />
            {active?.contribution_scopes?.map((value) => <Chip key={value} size="small" variant="outlined" color="primary" label={`已共享 · ${value}`} />)}
          </Box>
        </Box>
      </DialogTitle>
      <DialogContent sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "minmax(0,1fr) 300px" }, gap: { xs: 2, md: 3 }, p: { xs: 2, md: 3 }, overflow: "auto" }}>
        <Box sx={{ minWidth: 0, display: "grid", gridTemplateRows: "auto minmax(0,1fr)", gap: 1 }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
            <Typography variant="subtitle2" color="text.primary">影像预览</Typography>
            <FormControlLabel sx={{ mr: 0 }} control={<Switch size="small" checked={showLabelOverlay} disabled={active?.overlay_available === false} onChange={(_, checked) => setShowLabelOverlay(checked)} />} label="显示水体标注" />
          </Box>
          <Box sx={{ minHeight: { xs: 280, md: 500 }, display: "grid", placeItems: "center", border: 1, borderColor: "divider", bgcolor: "#111", overflow: "hidden" }}>
            {activePreviewUrl ? <Box component="img" src={activePreviewUrl} alt="" sx={{ display: "block", width: "100%", height: "100%", maxHeight: "62vh", objectFit: "contain" }} /> : <Typography color="grey.400">暂无预览</Typography>}
          </Box>
        </Box>
        <Box sx={{ minWidth: 0, display: "grid", alignContent: "start", gap: 2, borderLeft: { md: 1 }, borderColor: "divider", pl: { md: 3 } }}>
          <Box sx={{ display: "grid", gap: .75 }}>
            <Typography variant="overline" color="text.secondary" sx={{ lineHeight: 1.5 }}>共享状态</Typography>
            {active?.contribution_scopes?.length ? <Box sx={{ display: "flex", gap: .5, flexWrap: "wrap" }}>{active.contribution_scopes.map((value) => <Chip key={value} size="small" variant="outlined" color="primary" label={value} />)}</Box> : <Typography variant="body2" color="text.secondary">尚未共享</Typography>}
          </Box>
          <Box sx={{ display: "grid", gap: 1 }}>
            <Typography variant="overline" color="text.secondary" sx={{ lineHeight: 1.5 }}>影像与标注</Typography>
            {[['标注来源', patchSources(active)], ['水体像元', `${Number(active?.water_pixels || 0).toLocaleString()} px (${percent(active?.water_ratio_valid)})`], ['有效像元比例', percent(active?.valid_ratio)], ['影像产品', active?.product_name || '-'], ['来源记录', active?.sample_id || '-']].map(([label, value]) => <Box key={label} sx={{ minWidth: 0 }}><Typography variant="caption" color="text.secondary" display="block">{label}</Typography><Typography variant="body2" color="text.primary" sx={{ wordBreak: "break-word" }}>{value}</Typography></Box>)}
          </Box>
          <Box sx={{ display: "grid", gap: .75 }}>
            <Typography variant="overline" color="text.secondary" sx={{ lineHeight: 1.5 }}>Patch 标识</Typography>
            <Typography variant="body2" color="text.primary" sx={{ wordBreak: "break-all" }}>{active ? patchId(active) : "-"}</Typography>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: { xs: 2, md: 3 }, py: 1.5, borderTop: 1, borderColor: "divider", gap: .75, flexWrap: "wrap" }}>
        <Button disabled={busy} startIcon={active?.included ? <X size={15} /> : <RotateCcw size={15} />} onClick={() => active && update.mutate({ patch: active })}>{active?.included ? "排除" : "纳入"}</Button>
        <Button disabled={busy || !active?.included || activeShared} startIcon={<Share2 size={15} />} onClick={() => active && contributePatch(active)}>{activeShared ? "已共享" : "共享"}</Button>
        <Box sx={{ flex: 1 }} />
        <Button variant="outlined" onClick={() => setActive(null)}>关闭</Button>
      </DialogActions>
    </Dialog>
  </Box>;
}
