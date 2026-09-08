import { useState, type MouseEvent, type ReactNode } from "react";
import { Box, Button, Checkbox, FormControl, FormControlLabel, IconButton, InputLabel, Menu, MenuItem, Select, Slider, Tooltip, Typography } from "@mui/material";
import { Focus, Grid2X2, Layers3 } from "lucide-react";
import type { BasemapType, SiteLayerVisibility } from "./SiteMap";

interface SiteMapToolbarProps {
  leading?: ReactNode;
  imageryControl?: ReactNode;
  title: string;
  subtitle: string;
  basemap: BasemapType;
  onBasemapChange: (basemap: BasemapType) => void;
  visibility: SiteLayerVisibility;
  onVisibilityChange: (layer: keyof SiteLayerVisibility, visible: boolean) => void;
  jrcThreshold: number;
  onJrcThresholdChange: (threshold: number) => void;
  trainingAction?: ReactNode;
  showPrediction?: boolean;
  showPatches?: boolean;
  patchReviewEnabled?: boolean;
  onPatchReviewEnabledChange?: (enabled: boolean) => void;
  canFitSite?: boolean;
  canFitTile?: boolean;
  onFitSite: () => void;
  onFitTile: () => void;
}

const controlLabelSx = { m: 0, whiteSpace: "nowrap", "& .MuiFormControlLabel-label": { fontSize: 14 } };

export function SiteMapToolbar({
  leading,
  imageryControl,
  title,
  subtitle,
  basemap,
  onBasemapChange,
  visibility,
  onVisibilityChange,
  jrcThreshold,
  onJrcThresholdChange,
  trainingAction,
  showPrediction = false,
  showPatches = false,
  patchReviewEnabled = false,
  onPatchReviewEnabledChange,
  canFitSite = false,
  canFitTile = false,
  onFitSite,
  onFitTile,
}: SiteMapToolbarProps) {
  const [referenceAnchor, setReferenceAnchor] = useState<HTMLElement | null>(null);
  const checkbox = (layer: keyof SiteLayerVisibility, label: string) => (
    <FormControlLabel
      key={layer}
      sx={controlLabelSx}
      control={<Checkbox size="small" checked={visibility[layer]} onChange={(_, checked) => onVisibilityChange(layer, checked)} />}
      label={label}
    />
  );
  const referenceLayerCount = [visibility.osm, visibility.hydro, visibility.context, visibility.esa, visibility.jrc].filter(Boolean).length;
  const openReferenceLayers = (event: MouseEvent<HTMLButtonElement>) => setReferenceAnchor(event.currentTarget);
  const referenceCheckbox = (layer: keyof SiteLayerVisibility, label: string) => (
    <MenuItem key={layer} dense disableRipple onClick={() => onVisibilityChange(layer, !visibility[layer])}>
      <Checkbox size="small" checked={visibility[layer]} tabIndex={-1} disableRipple slotProps={{ input: { "aria-label": label } }} />
      <Typography variant="body2">{label}</Typography>
    </MenuItem>
  );

  return (
    <Box data-testid="site-map-toolbar" sx={{ bgcolor: "background.paper", borderBottom: 1, borderColor: "divider", minWidth: 0 }}>
      <Box sx={{ minHeight: 64, px: { xs: 1, sm: 2 }, py: 1, display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
        {leading}
        <Box sx={{ minWidth: 150, flex: "1 1 240px" }}>
          <Typography variant="subtitle1" color="text.primary" noWrap title={title}>{title}</Typography>
          <Typography variant="caption" component="div" noWrap title={subtitle}>{subtitle}</Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: .5, flexWrap: "wrap" }}>
          <FormControl size="small" sx={{ minWidth: 125 }}>
            <InputLabel id="basemap-label">底图</InputLabel>
            <Select labelId="basemap-label" value={basemap} label="底图" onChange={(event) => onBasemapChange(event.target.value as BasemapType)}>
              <MenuItem value="osm">OSM 地图</MenuItem>
              <MenuItem value="satellite">卫星图</MenuItem>
              <MenuItem value="none">无</MenuItem>
            </Select>
          </FormControl>
          <Box sx={{ display: "flex", alignItems: "center", gap: .25 }}>
            {checkbox("image", "影像")}
            {imageryControl}
          </Box>
          <Tooltip title="定位观测区域"><span><IconButton size="small" disabled={!canFitSite} onClick={onFitSite} aria-label="定位观测区域"><Focus size={18} /></IconButton></span></Tooltip>
          <Tooltip title="定位 Tile"><span><IconButton size="small" disabled={!canFitTile} onClick={onFitTile} aria-label="定位 Tile"><Grid2X2 size={18} /></IconButton></span></Tooltip>
        </Box>
      </Box>
      <Box sx={{ px: { xs: 1, sm: 2 }, py: .5, borderTop: 1, borderColor: "divider", overflowX: "auto" }}>
        <Box sx={{ minWidth: "max-content", display: "flex", alignItems: "center", gap: .75 }}>
          {checkbox("tile", "Tile")}
          <Button size="small" variant="outlined" color="inherit" startIcon={<Layers3 size={16} />} onClick={openReferenceLayers} aria-haspopup="menu" aria-expanded={Boolean(referenceAnchor)}>
            参考图层{referenceLayerCount ? ` ${referenceLayerCount}` : ""}
          </Button>
          <Menu anchorEl={referenceAnchor} open={Boolean(referenceAnchor)} onClose={() => setReferenceAnchor(null)} slotProps={{ paper: { sx: { minWidth: 250 } } }}>
            {referenceCheckbox("osm", "OSM 水体（矢量）")}
            {referenceCheckbox("hydro", "HydroLAKES")}
            {referenceCheckbox("context", "其他水体")}
            {referenceCheckbox("esa", "ESA")}
            {referenceCheckbox("jrc", "JRC")}
            <Box sx={{ px: 2, pt: .5, pb: 1, display: "grid", gridTemplateColumns: "1fr 40px", alignItems: "center", gap: 1 }} onClick={(event) => event.stopPropagation()}>
              <Slider aria-label="JRC 阈值" size="small" min={1} max={100} value={jrcThreshold} onChangeCommitted={(_, value) => onJrcThresholdChange(value as number)} />
              <Typography variant="caption" textAlign="right">{jrcThreshold}%</Typography>
            </Box>
          </Menu>
          {checkbox("spectralWater", "光谱水体（当前影像）")}
          {checkbox("spectralOsmIntersection", "光谱 ∩ OSM（高置信种子）")}
          {checkbox("spectralOsmConsensus", "光谱主导 · OSM 约束")}
          {checkbox("osmSpectralConsensus", "OSM 主导 · 光谱候选")}
          {checkbox("local", "本体标注")}
          {showPrediction && checkbox("prediction", "模型预测")}
          {showPatches && <FormControlLabel sx={controlLabelSx} control={<Checkbox size="small" checked={patchReviewEnabled} onChange={(_, checked) => onPatchReviewEnabledChange?.(checked)} />} label="Patch" />}
          {trainingAction && <Box sx={{ display: "flex", alignItems: "center", ml: "auto" }}>{trainingAction}</Box>}
        </Box>
      </Box>
    </Box>
  );
}
