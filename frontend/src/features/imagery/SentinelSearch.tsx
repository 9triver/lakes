import { useMemo, useState } from "react";
import { Box, Button, CircularProgress, FormControl, InputLabel, MenuItem, Select, Slider, TextField, Typography } from "@mui/material";
import { CloudDownload, Search } from "lucide-react";
import type { SentinelProduct } from "../../api/types";
import { searchSentinelProducts, useDownloadSentinel, useSentinelTiles } from "./api";

function isoDate(date: Date) { return date.toISOString().slice(0, 10); }

export function SentinelSearch({ region, siteId }: { region: string; siteId: string }) {
  const now = useMemo(() => new Date(), []);
  const initialStart = useMemo(() => { const date = new Date(now); date.setMonth(date.getMonth() - 2); return isoDate(date); }, [now]);
  const tiles = useSentinelTiles(region, siteId);
  const download = useDownloadSentinel(region, siteId);
  const [tile, setTile] = useState("");
  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(isoDate(now));
  const [cloud, setCloud] = useState(50);
  const [products, setProducts] = useState<SentinelProduct[]>([]);
  const [searching, setSearching] = useState(false);
  const selectedTile = tile || tiles.data?.[0]?.tile || "";

  async function search() {
    if (!selectedTile) return;
    setSearching(true);
    try { setProducts(await searchSentinelProducts(region, siteId, { tile: selectedTile, start, end, cloud })); }
    finally { setSearching(false); }
  }

  return (
    <Box component="section" sx={{ width: "100%", borderTop: 1, borderColor: "divider", pt: 1, display: "grid", gap: 1 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        <FormControl sx={{ minWidth: 110 }}><InputLabel>Tile</InputLabel><Select label="Tile" value={selectedTile} onChange={(event) => setTile(event.target.value)}>{(tiles.data || []).map((item) => <MenuItem key={item.tile} value={item.tile}>{item.tile}{item.downloaded ? " · 已下载" : ""}</MenuItem>)}</Select></FormControl>
        <TextField type="date" label="开始" value={start} onChange={(event) => setStart(event.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
        <TextField type="date" label="结束" value={end} onChange={(event) => setEnd(event.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
        <Box sx={{ width: 150, display: "flex", alignItems: "center", gap: 1 }}><Slider size="small" min={0} max={100} value={cloud} onChange={(_, value) => setCloud(value as number)} /><Typography variant="caption">云量 {cloud}%</Typography></Box>
        <Button variant="outlined" startIcon={searching ? <CircularProgress size={15} /> : <Search size={16} />} disabled={!selectedTile || searching} onClick={search}>查询产品</Button>
      </Box>
      {products.length > 0 && <Box sx={{ maxHeight: 180, overflow: "auto", borderTop: 1, borderColor: "divider" }}>
        {products.map((product) => <Box key={product.product_id || product.name} sx={{ minHeight: 42, display: "grid", gridTemplateColumns: "90px 80px 100px minmax(160px,1fr) auto", alignItems: "center", gap: 1, borderBottom: 1, borderColor: "divider", fontSize: 13 }}>
          <span>{product.date || ""}</span><span>云量 {Number(product.cloud_cover || 0).toFixed(1)}%</span><span>覆盖 {(Number(product.site_coverage_ratio || 0) * 100).toFixed(0)}%</span><Typography variant="caption" noWrap title={product.name}>{product.name}</Typography>
          <Button startIcon={<CloudDownload size={15} />} disabled={product.downloaded || download.isPending} onClick={() => download.mutate(product)}>{product.downloaded ? "已下载" : download.isPending ? "下载中" : "下载"}</Button>
        </Box>)}
      </Box>}
      {!searching && products.length === 0 && <Typography variant="caption" color="text.secondary">选择 tile 和日期范围后查询</Typography>}
      {download.isError && <Typography variant="caption" color="error">{download.error.message}</Typography>}
    </Box>
  );
}
