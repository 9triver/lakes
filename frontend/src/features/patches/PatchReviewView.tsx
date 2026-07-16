import { useMemo, useState } from "react";
import { Box, Button, Chip, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Pagination, Select, Typography } from "@mui/material";
import { Eye, LocateFixed, RotateCcw, X } from "lucide-react";
import type { TrainingPatch } from "../../api/types";
import { useTrainingPatches, useUpdateTrainingPatch } from "./api";
import { PatchGenerationControls } from "./PatchGenerationControls";

function patchSiteName(patch?: TrainingPatch | null) {
  return patch?.site_display_name || patch?.site_name || patch?.site_id || "Patch 预览";
}

function patchSources(patch?: TrainingPatch | null) {
  return (patch?.label_sources || patch?.label_source || "未记录").split(",").filter(Boolean).join(" + ");
}

function percent(value?: number) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export function PatchReviewView({ scope, onLocate }: { scope: string; onLocate: (patch: TrainingPatch) => void }) {
  const [include, setInclude] = useState("");
  const [water, setWater] = useState("");
  const [page, setPage] = useState(1);
  const [active, setActive] = useState<TrainingPatch | null>(null);
  const query = useTrainingPatches(scope, include);
  const update = useUpdateTrainingPatch(scope, include);
  const filtered = useMemo(
    () => (query.data?.items || []).filter((patch) => water === "water" ? Number(patch.water_pixels || 0) > 0 : water === "negative" ? Number(patch.water_pixels || 0) <= 0 : true),
    [query.data?.items, water],
  );
  const pageSize = 18;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const items = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return <Box sx={{ height: "100%", overflow: "auto" }}>
    <PatchGenerationControls scope={scope} />
    <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <Typography variant="subtitle1" sx={{ mr: "auto" }}>逻辑 Patch 审核 · {filtered.length}</Typography>
      <FormControl sx={{ minWidth: 120 }}><InputLabel>状态</InputLabel><Select label="状态" value={include} onChange={(event) => { setInclude(event.target.value); setPage(1); }}><MenuItem value="">全部</MenuItem><MenuItem value="included">包含</MenuItem><MenuItem value="excluded">排除</MenuItem></Select></FormControl>
      <FormControl sx={{ minWidth: 130 }}><InputLabel>水体像元</InputLabel><Select label="水体像元" value={water} onChange={(event) => { setWater(event.target.value); setPage(1); }}><MenuItem value="">全部</MenuItem><MenuItem value="water">有水体</MenuItem><MenuItem value="negative">无水体</MenuItem></Select></FormControl>
    </Box>
    {query.isLoading ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 240 }}><CircularProgress size={28} /></Box> : query.isError ? <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography> : items.length ? <>
      <Box sx={{ p: 1.5, display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(210px,1fr))", gap: 1 }}>
        {items.map((patch) => <Box key={`${patch.region || scope}:${patch.patch_id}`} sx={{ border: 1, borderColor: patch.included ? "divider" : "error.light", bgcolor: "background.paper", opacity: patch.included ? 1 : .65 }}>
          <Box component="button" onClick={() => setActive(patch)} sx={{ border: 0, p: 0, width: "100%", aspectRatio: "1", bgcolor: "#111", cursor: "pointer" }}>{patch.preview_url && <Box component="img" src={patch.preview_url} alt="" loading="lazy" sx={{ width: "100%", height: "100%", objectFit: "contain" }} />}</Box>
          <Box sx={{ p: 1, display: "grid", gap: .35 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: .5 }}><Typography noWrap variant="body2" sx={{ flex: 1 }}>{patchSiteName(patch)}</Typography><Chip size="small" color={patch.included ? "success" : "error"} variant="outlined" label={patch.included ? "包含" : "已排除"} /></Box>
            <Typography variant="caption" color="text.secondary" noWrap title={patchSources(patch)}>标注 {patchSources(patch)}</Typography>
            <Typography variant="caption" color="text.secondary">水体 {Number(patch.water_pixels || 0).toLocaleString()} px · {percent(patch.water_ratio_valid)} · 有效 {percent(patch.valid_ratio)}</Typography>
            <Typography variant="caption" color="text.secondary" noWrap title={patch.product_name}>{patch.product_date || patch.product_name || patch.sample_id}</Typography>
            <Box sx={{ display: "flex", mt: .25 }}><Button disabled={update.isPending} startIcon={patch.included ? <X size={14} /> : <RotateCcw size={14} />} onClick={() => update.mutate(patch)}>{patch.included ? "排除" : "恢复"}</Button><Button startIcon={<LocateFixed size={14} />} onClick={() => onLocate(patch)}>定位</Button></Box>
          </Box>
        </Box>)}
      </Box>
      <Box sx={{ display: "flex", justifyContent: "center", py: 1 }}><Pagination count={pageCount} page={currentPage} onChange={(_, value) => setPage(value)} /></Box>
    </> : <Box sx={{ p: 4, textAlign: "center" }}><Typography>当前筛选条件下没有 Patch</Typography><Typography variant="caption" color="text.secondary">可使用上方控件从训练样本生成</Typography></Box>}
    <Dialog open={Boolean(active)} onClose={() => setActive(null)} maxWidth="lg" fullWidth><DialogTitle>{patchSiteName(active)}</DialogTitle><DialogContent sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "minmax(0,1fr) 320px" }, gap: 2 }}>
      {active?.preview_url && <Box component="img" src={active.preview_url} alt="" sx={{ width: "100%", maxHeight: "70vh", objectFit: "contain", bgcolor: "#111" }} />}
      <Box sx={{ display: "grid", gap: 1, alignContent: "start", minWidth: 0 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}><Chip size="small" color={active?.included ? "success" : "error"} label={active?.included ? "包含在逻辑训练集" : "已从逻辑训练集排除"} /><Box sx={{ width: 14, height: 14, bgcolor: "#ff2080" }} /><Typography variant="caption">水体掩膜</Typography></Box>
        {[['标注来源并集', patchSources(active)], ['水体像元', `${Number(active?.water_pixels || 0).toLocaleString()} px (${percent(active?.water_ratio_valid)})`], ['有效像元比例', percent(active?.valid_ratio)], ['影像日期', active?.product_date || '-'], ['影像产品', active?.product_name || '-'], ['训练样本', active?.sample_id || '-'], ['逻辑 Patch', active?.logical_patch_id || active?.patch_id || '-']].map(([label, value]) => <Box key={label} sx={{ borderBottom: 1, borderColor: "divider", pb: .75 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" color="text.primary" sx={{ wordBreak: "break-all" }}>{value}</Typography></Box>)}
      </Box>
    </DialogContent><DialogActions><Button startIcon={<Eye size={15} />} onClick={() => active && update.mutate(active)}>{active?.included ? "排除" : "恢复包含"}</Button><Button onClick={() => setActive(null)}>关闭</Button></DialogActions></Dialog>
  </Box>;
}
