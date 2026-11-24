import { Box, Typography } from "@mui/material";
import type { RoutingInfo } from "./api";

export function RoutingSummary({ routing, preview }: { routing?: RoutingInfo | null; preview?: string }) {
  if (!routing) return null;
  return <Box component="details" data-testid="routing-summary" sx={{ p: 1, bgcolor: "background.paper", border: 1, borderColor: "divider", borderRadius: 1 }}>
    <Box component="summary" sx={{ cursor: "pointer", fontSize: 13, fontWeight: 600 }}>专家调用与路由图 · {routing.windows}个窗口 · Global回退 {(routing.fallback_fraction * 100).toFixed(2)}%</Box>
    <Typography variant="caption" color="text.secondary" display="block" sx={{ my: 1 }}>覆盖比例按有效像元的融合权重统计；地图颜色表示重叠窗口中占主导的模型。</Typography>
    {Boolean(routing.rejected_windows) && <Typography variant="caption" display="block">原K4的C3拒绝窗口：{routing.rejected_windows}个，已回退Global。</Typography>}
    {routing.components.map((row) => <Box key={row.cluster} sx={{ display: "flex", gap: 1, alignItems: "center", mb: .5 }}>
      <Box component="span" sx={{ width: 12, height: 12, bgcolor: row.color, flexShrink: 0 }} />
      <Typography variant="caption">{row.label}：{row.windows}个窗口 · 覆盖 {(row.coverage_fraction * 100).toFixed(2)}%</Typography>
    </Box>)}
    <Box sx={{ position: "relative", maxWidth: 768, mt: 1 }}>
      {preview && <img src={preview} alt="路由图底图" style={{ width: "100%", display: "block" }} />}
      <img src={routing.overlay} alt={`K${routing.k ?? routing.components.length - 1}专家路由分布图`} style={{ width: "100%", display: "block", ...(preview ? { position: "absolute" as const, inset: 0, height: "100%" } : {}) }} />
    </Box>
    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>{routing.rule || "有效影像比例不足80%的窗口回退Global；其余按最近聚类中心选择专家。"}</Typography>
  </Box>;
}
