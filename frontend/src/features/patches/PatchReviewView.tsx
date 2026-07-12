import { useMemo, useState } from "react";
import { Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Pagination, Select, Typography } from "@mui/material";
import { Eye, LocateFixed, RotateCcw, X } from "lucide-react";
import type { TrainingPatch } from "../../api/types";
import { useTrainingPatches, useUpdateTrainingPatch } from "./api";
import { PatchGenerationControls } from "./PatchGenerationControls";

export function PatchReviewView({ scope, onLocate }: { scope: string; onLocate: (patch: TrainingPatch) => void }) {
  const [include, setInclude] = useState("");
  const [water, setWater] = useState("");
  const [page, setPage] = useState(1);
  const [active, setActive] = useState<TrainingPatch | null>(null);
  const query = useTrainingPatches(scope, include);
  const update = useUpdateTrainingPatch(scope, include);
  const filtered = useMemo(() => (query.data?.items || []).filter((patch) => water === "water" ? Number(patch.water_pixels || 0) > 0 : water === "negative" ? Number(patch.water_pixels || 0) <= 0 : true), [query.data?.items, water]);
  const pageSize = 18;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const items = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  return <Box sx={{ height: "100%", overflow: "auto" }}>
    <PatchGenerationControls scope={scope} />
    <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}><Typography variant="subtitle1" sx={{ mr: "auto" }}>Patch 审核 · {filtered.length}</Typography><FormControl sx={{ minWidth: 120 }}><InputLabel>状态</InputLabel><Select label="状态" value={include} onChange={(event) => { setInclude(event.target.value); setPage(1); }}><MenuItem value="">全部</MenuItem><MenuItem value="included">包含</MenuItem><MenuItem value="excluded">排除</MenuItem></Select></FormControl><FormControl sx={{ minWidth: 130 }}><InputLabel>水体像元</InputLabel><Select label="水体像元" value={water} onChange={(event) => { setWater(event.target.value); setPage(1); }}><MenuItem value="">全部</MenuItem><MenuItem value="water">有水体</MenuItem><MenuItem value="negative">无水体</MenuItem></Select></FormControl></Box>
    {query.isLoading ? <Box sx={{ display: "grid", placeItems: "center", minHeight: 240 }}><CircularProgress size={28} /></Box> : query.isError ? <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography> : items.length ? <>
      <Box sx={{ p: 1.5, display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(210px,1fr))", gap: 1 }}>{items.map((patch) => <Box key={`${patch.region || scope}:${patch.patch_id}`} sx={{ border: 1, borderColor: patch.included ? "divider" : "error.light", bgcolor: "background.paper", opacity: patch.included ? 1 : .65 }}><Box component="button" onClick={() => setActive(patch)} sx={{ border: 0, p: 0, width: "100%", aspectRatio: "1", bgcolor: "#111", cursor: "pointer" }}>{patch.preview_url && <Box component="img" src={patch.preview_url} alt="" loading="lazy" sx={{ width: "100%", height: "100%", objectFit: "contain" }} />}</Box><Box sx={{ p: 1 }}><Typography noWrap variant="body2">{patch.lake_display_name || patch.lake_name || patch.lake_id}</Typography><Typography variant="caption" color="text.secondary">water {(Number(patch.water_ratio_valid || 0) * 100).toFixed(1)}% · valid {(Number(patch.valid_ratio || 0) * 100).toFixed(1)}%</Typography><Box sx={{ display: "flex", mt: .5 }}><Button disabled={update.isPending} startIcon={patch.included ? <X size={14} /> : <RotateCcw size={14} />} onClick={() => update.mutate(patch)}>{patch.included ? "排除" : "恢复"}</Button><Button startIcon={<LocateFixed size={14} />} onClick={() => onLocate(patch)}>定位</Button></Box></Box></Box>)}</Box>
      <Box sx={{ display: "flex", justifyContent: "center", py: 1 }}><Pagination count={pageCount} page={currentPage} onChange={(_, value) => setPage(value)} /></Box>
    </> : <Box sx={{ p: 4, textAlign: "center" }}><Typography>当前筛选条件下没有 Patch</Typography><Typography variant="caption" color="text.secondary">可使用上方控件从训练样本生成</Typography></Box>}
    <Dialog open={Boolean(active)} onClose={() => setActive(null)} maxWidth="lg"><DialogTitle>{active?.lake_display_name || active?.lake_name || "Patch 预览"}</DialogTitle><DialogContent>{active?.preview_url && <Box component="img" src={active.preview_url} alt="" sx={{ maxWidth: "80vw", maxHeight: "70vh", objectFit: "contain" }} />}<Typography variant="caption" display="block">{active?.patch_id}</Typography></DialogContent><DialogActions><Button startIcon={<Eye size={15} />} onClick={() => active && update.mutate(active)}>{active?.included ? "排除" : "恢复包含"}</Button><Button onClick={() => setActive(null)}>关闭</Button></DialogActions></Dialog>
  </Box>;
}
