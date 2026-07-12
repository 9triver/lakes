import { useEffect, useState } from "react";
import { Box, Button, Chip, CircularProgress, MenuItem, Select, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { LocateFixed, Save, Trash2 } from "lucide-react";
import type { TrainingSample } from "../../api/types";
import { useDeleteTrainingSample, useTrainingSamples, useUpdateTrainingSample } from "./api";

function SampleRow({ sample, scope, onLocate }: { sample: TrainingSample; scope: string; onLocate: (sample: TrainingSample) => void }) {
  const [split, setSplit] = useState(sample.split || "");
  const [notes, setNotes] = useState(sample.notes || "");
  const update = useUpdateTrainingSample(scope);
  const remove = useDeleteTrainingSample(scope);
  useEffect(() => { setSplit(sample.split || ""); setNotes(sample.notes || ""); }, [sample.notes, sample.split]);
  return <TableRow hover>
    <TableCell><Typography variant="body2">{sample.lake_display_name || sample.lake_name || sample.lake_id}</Typography><Typography variant="caption" color="text.secondary">{sample.sample_id}</Typography></TableCell>
    <TableCell>{sample.region_name || sample.region || scope}</TableCell>
    <TableCell><Chip size="small" color={sample.status === "ok" ? "success" : "warning"} label={sample.status === "ok" ? "完整" : "缺文件"} /></TableCell>
    <TableCell>{sample.label_source || "current_view"}</TableCell>
    <TableCell>{sample.tile_count || 0} · {sample.product_date || ""}</TableCell>
    <TableCell><Select size="small" value={split} onChange={(event) => setSplit(event.target.value)} sx={{ minWidth: 88 }}>{["", "train", "val", "test"].map((value) => <MenuItem key={value} value={value}>{value || "unsplit"}</MenuItem>)}</Select></TableCell>
    <TableCell><TextField value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="备注" /></TableCell>
    <TableCell><Box sx={{ display: "flex", gap: .5 }}><Button startIcon={<LocateFixed size={15} />} onClick={() => onLocate(sample)}>定位</Button><Button startIcon={<Save size={15} />} disabled={update.isPending} onClick={() => update.mutate({ sample, changes: { split, notes } })}>保存</Button><Button color="error" startIcon={<Trash2 size={15} />} disabled={remove.isPending} onClick={() => { if (confirm("删除这个训练样本记录？")) remove.mutate(sample); }}>删除</Button></Box></TableCell>
  </TableRow>;
}

export function TrainingSamplesView({ scope, onLocate }: { scope: string; onLocate: (sample: TrainingSample) => void }) {
  const query = useTrainingSamples(scope);
  if (query.isLoading) return <Box sx={{ display: "grid", placeItems: "center", height: "100%" }}><CircularProgress size={28} /></Box>;
  if (query.isError) return <Typography color="error" sx={{ p: 2 }}>{query.error.message}</Typography>;
  return <Box sx={{ height: "100%", overflow: "auto" }}>
    <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: "divider" }}><Typography variant="subtitle1">训练样本</Typography><Typography variant="caption" color="text.secondary">{query.data?.total || 0} 个样本</Typography></Box>
    {(query.data?.items || []).length ? <Box sx={{ overflowX: "auto" }}><Table size="small" stickyHeader><TableHead><TableRow><TableCell>水体</TableCell><TableCell>区域</TableCell><TableCell>状态</TableCell><TableCell>标注</TableCell><TableCell>影像</TableCell><TableCell>Split</TableCell><TableCell>备注</TableCell><TableCell>操作</TableCell></TableRow></TableHead><TableBody>{(query.data?.items || []).map((sample) => <SampleRow key={`${sample.region || scope}:${sample.sample_id}`} sample={sample} scope={scope} onLocate={onLocate} />)}</TableBody></Table></Box> : <Box sx={{ p: 4, textAlign: "center" }}><Typography>当前区域还没有训练样本</Typography><Typography variant="caption" color="text.secondary">从湖泊页面记录当前视图后会显示在这里</Typography></Box>}
  </Box>;
}
