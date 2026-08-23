import { useEffect, useState } from "react";
import { Box, FormControl, InputLabel, MenuItem, Select, Typography } from "@mui/material";
import type { ImageryProduct } from "../../api/types";
import { useImagery, useSetActiveImagery } from "./api";

export interface ImagerySelection {
  assetId: string;
  product: string;
  tile: string;
  localLabelId: string;
  localLabel: ImageryProduct["label"];
}

export function ImageryPanel({ region, siteId, onSelectionChange, compact = false }: { region: string; siteId: string; onSelectionChange?: (selection: ImagerySelection) => void; compact?: boolean }) {
  const imagery = useImagery(region, siteId);
  const setActive = useSetActiveImagery(region, siteId);
  const [assetId, setAssetId] = useState("");
  const assets = imagery.data?.assets || [];
  const selected = assets.find((item) => (item.asset_id || item.product) === assetId) || assets.find((item) => item.active);

  useEffect(() => {
    const next = assets.find((item) => item.active) || assets[0];
    const nextId = next ? (next.asset_id || next.product) : "";
    if (nextId !== assetId) setAssetId(nextId);
  }, [assetId, assets]);
  useEffect(() => {
    if (!selected) return;
    onSelectionChange?.({ assetId: selected.asset_id || selected.product, product: selected.product, tile: selected.tile || "", localLabelId: selected.label_id || "", localLabel: selected.label || null });
  }, [onSelectionChange, selected]);

  const selectAsset = (nextId: string) => {
    const next = assets.find((item) => (item.asset_id || item.product) === nextId);
    if (!next) return;
    setAssetId(nextId);
    setActive.mutate({ assetId: next.asset_id || next.product, product: next.product });
  };

  return (
    <Box sx={{ px: compact ? 0 : 2, py: compact ? 0 : 1, bgcolor: "background.paper", borderTop: compact ? 0 : 1, borderColor: "divider", display: "flex", alignItems: "center", gap: compact ? .5 : 1, flexWrap: "wrap", minWidth: 0 }}>
      <FormControl size="small" sx={{ minWidth: compact ? 220 : 280, flex: compact ? "0 1 300px" : 1 }}>
        <InputLabel>本地影像期次</InputLabel>
        <Select label="本地影像期次" value={selected ? (selected.asset_id || selected.product) : ""} onChange={(event) => selectAsset(event.target.value)} inputProps={{ "aria-label": "本地影像期次" }}>
          {assets.map((item: ImageryProduct) => <MenuItem key={item.asset_id || item.product} value={item.asset_id || item.product}>{item.active ? "当前 · " : ""}{item.date || "未知日期"} · {item.label ? "有同期标注" : "无同期标注"}</MenuItem>)}
        </Select>
      </FormControl>
      {setActive.isPending && <Typography variant="caption">切换中</Typography>}
      {setActive.isError && <Typography variant="caption" color="error">{setActive.error.message}</Typography>}
      {!assets.length && <Typography variant="caption" color="text.secondary">没有可用的本地影像</Typography>}
    </Box>
  );
}
