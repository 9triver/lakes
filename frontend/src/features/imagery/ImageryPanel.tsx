import { useEffect, useMemo, useState } from "react";
import { Box, Button, FormControl, InputLabel, MenuItem, Select, Typography } from "@mui/material";
import { Image } from "lucide-react";
import { useImagery, useSetActiveImagery } from "./api";
import { SentinelSearch } from "./SentinelSearch";

export function ImageryPanel({ region, siteId, onSelectionChange }: { region: string; siteId: string; onSelectionChange?: (selection: { tile: string; product: string }) => void }) {
  const imagery = useImagery(region, siteId);
  const setActive = useSetActiveImagery(region, siteId);
  const [tile, setTile] = useState("");
  const [product, setProduct] = useState("");
  const tiles = imagery.data?.tiles || [];
  const products = useMemo(() => tiles.find((item) => item.tile === tile)?.products || [], [tile, tiles]);

  useEffect(() => {
    if (!tiles.some((item) => item.tile === tile)) setTile(tiles[0]?.tile || "");
  }, [tile, tiles]);
  useEffect(() => {
    if (!products.some((item) => item.product === product)) setProduct(products.find((item) => item.active)?.product || products[0]?.product || "");
  }, [product, products]);
  useEffect(() => { onSelectionChange?.({ tile, product }); }, [onSelectionChange, product, tile]);

  return (
    <Box sx={{ px: 2, py: 1, bgcolor: "background.paper", borderTop: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
      <FormControl sx={{ minWidth: 115 }}><InputLabel>Tile</InputLabel><Select label="Tile" value={tile} onChange={(event) => setTile(event.target.value)}>{tiles.map((item) => <MenuItem key={item.tile} value={item.tile}>{item.tile}</MenuItem>)}</Select></FormControl>
      <FormControl sx={{ minWidth: 280, flex: 1 }}><InputLabel>影像产品</InputLabel><Select label="影像产品" value={product} onChange={(event) => setProduct(event.target.value)}>{products.map((item) => <MenuItem key={item.product} value={item.product}>{item.active ? "当前 · " : ""}{item.asset_label || item.source || "影像"} · {item.date || item.product}</MenuItem>)}</Select></FormControl>
      <Button variant="contained" startIcon={<Image size={16} />} disabled={!tile || !product || setActive.isPending} onClick={() => setActive.mutate({ tile, product })}>设为影像</Button>
      <Typography variant="caption" color={setActive.isError ? "error" : "text.secondary"}>{setActive.isPending ? "切换中" : setActive.isSuccess ? "影像已更新" : `${products.length} 个候选`}</Typography>
      <SentinelSearch region={region} siteId={siteId} />
    </Box>
  );
}
