import { Box, Button, Chip, FormControl, InputLabel, MenuItem, Select, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { RotateCcw, X } from "lucide-react";
import type { TrainingPatch } from "../../api/types";

export type PatchOperation = "exclude" | "restore";

export interface PatchGroup {
  key: string;
  label: string;
  patches: TrainingPatch[];
}

export function SitePatchReviewPanel({ groups, groupKey, onGroupChange, operation, onOperationChange, active, pendingCount, onApply, onClear, applying }: {
  groups: PatchGroup[];
  groupKey: string;
  onGroupChange: (value: string) => void;
  operation: PatchOperation;
  onOperationChange: (value: PatchOperation) => void;
  active?: TrainingPatch;
  pendingCount: number;
  onApply: () => void;
  onClear: () => void;
  applying: boolean;
}) {
  return <Box sx={{ borderTop: 1, borderColor: "divider", bgcolor: "background.paper", p: 1.25, display: "grid", gridTemplateColumns: { xs: "1fr", sm: "minmax(280px,1fr) minmax(260px,420px)" }, gap: 1.25, maxHeight: { xs: "42vh", sm: 210 }, overflow: "auto" }}>
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", alignContent: "start" }}>
      <FormControl size="small" sx={{ minWidth: 250, flex: 1 }}><InputLabel id="patch-group-label">来源影像</InputLabel><Select labelId="patch-group-label" label="来源影像" value={groupKey} onChange={(event) => onGroupChange(event.target.value)}>{groups.map((group) => <MenuItem key={group.key} value={group.key}>{group.label}</MenuItem>)}</Select></FormControl>
      <ToggleButtonGroup exclusive size="small" value={operation} onChange={(_, value: PatchOperation | null) => value && onOperationChange(value)} aria-label="Patch 操作">
        <ToggleButton value="exclude"><X size={15} />排除</ToggleButton>
        <ToggleButton value="restore"><RotateCcw size={15} />恢复</ToggleButton>
      </ToggleButtonGroup>
      <Chip size="small" color={pendingCount ? "warning" : "default"} label={`待处理 ${pendingCount}`} />
      <Button variant="contained" disabled={!pendingCount || applying} onClick={onApply}>应用</Button>
      <Button disabled={!pendingCount || applying} onClick={onClear}>清空</Button>
      <Typography variant="caption" color="text.secondary" sx={{ width: "100%" }}>单击网格选择；绿色为包含，红色为已排除，黄色为待处理。</Typography>
    </Box>
    <Box sx={{ display: "grid", gridTemplateColumns: "128px minmax(0,1fr)", gap: 1, minWidth: 0 }}>
      <Box sx={{ width: 128, height: 128, bgcolor: "#111", display: "grid", placeItems: "center" }}>{active?.preview_url ? <Box component="img" src={active.preview_url} alt="" sx={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <Typography variant="caption" color="grey.400">选择一个 Patch</Typography>}</Box>
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="subtitle2" color="text.primary" noWrap>{active?.included ? "已包含" : active ? "已排除" : "Patch 详情"}</Typography>
        {active && <>
          <Typography variant="body2">水体 {(Number(active.water_ratio_valid || 0) * 100).toFixed(1)}%</Typography>
          <Typography variant="body2">水体像元 {Number(active.water_pixels || 0).toLocaleString()}</Typography>
          <Typography variant="body2">有效 {(Number(active.valid_ratio || 0) * 100).toFixed(1)}%</Typography>
          <Typography variant="caption" display="block" noWrap title={active.label_sources || active.label_source}>标注 {(active.label_sources || active.label_source || "未记录").split(",").join(" + ")}</Typography>
          <Typography variant="caption" display="block" noWrap title={active.product_name}>{active.product_date || active.product_name || "-"}</Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ wordBreak: "break-all" }}>{active.logical_patch_id || active.patch_id}</Typography>
        </>}
      </Box>
    </Box>
  </Box>;
}
