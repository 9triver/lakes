import { useEffect, useState } from "react";
import { Alert, Box, Button, Checkbox, CircularProgress, FormControlLabel, Typography } from "@mui/material";
import { Check } from "lucide-react";
import { useResolveSourceConflict, useSourceConflicts } from "./api";

function ConflictRow({ profileId, item }: { profileId: string; item: NonNullable<ReturnType<typeof useSourceConflicts>["data"]>["items"][number] }) {
  const [selected, setSelected] = useState<string[]>(item.candidate_variant_ids);
  const resolve = useResolveSourceConflict(profileId);
  useEffect(() => setSelected(item.candidate_variant_ids), [item.candidate_variant_ids]);
  return <Box sx={{ borderBottom: 1, borderColor: "divider", p: 2, display: "grid", gap: 1 }}>
    <Typography variant="subtitle2" color="text.primary">{item.site_id}</Typography>
    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>{item.variants.map((variant) => <FormControlLabel key={variant.id} control={<Checkbox checked={selected.includes(variant.id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, variant.id] : current.filter((value) => value !== variant.id))} />} label={variant.label} />)}</Box>
    {resolve.isError && <Alert severity="error">{resolve.error.message}</Alert>}
    <Button sx={{ justifySelf: "start" }} variant="contained" startIcon={<Check size={15} />} disabled={!selected.length || resolve.isPending} onClick={() => resolve.mutate({ siteId: item.site_id, variantIds: selected })}>确认来源</Button>
  </Box>;
}

export function SourceConflictsView({ profileId }: { profileId: string }) {
  const query = useSourceConflicts(profileId);
  if (query.isLoading) return <Box sx={{ display: "grid", placeItems: "center", minHeight: 240 }}><CircularProgress size={28} /></Box>;
  if (query.isError) return <Alert severity="error" sx={{ m: 2 }}>{query.error.message}</Alert>;
  if (!query.data?.items.length) return <Box sx={{ p: 4, textAlign: "center" }}><Typography color="text.primary">没有待处理的来源冲突</Typography></Box>;
  return <Box sx={{ height: "100%", overflow: "auto" }}><Alert severity="warning" sx={{ borderRadius: 0 }}>{query.data.total} 个 site 需要确认来源</Alert>{query.data.items.map((item) => <ConflictRow key={item.site_id} profileId={profileId} item={item} />)}</Box>;
}
