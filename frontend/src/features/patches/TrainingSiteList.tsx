import type { ReactNode } from "react";
import { Box, Chip, CircularProgress, List, ListItemButton, ListItemText, Typography } from "@mui/material";
import { useTrainingPatchSites } from "./api";

interface TrainingSiteListProps {
  workspaceId: string;
  scope: string;
  selectedSiteId: string;
  regionControl: ReactNode;
  onSelect: (siteId: string) => void;
}

export function TrainingSiteList({ workspaceId, scope, selectedSiteId, regionControl, onSelect }: TrainingSiteListProps) {
  const sites = useTrainingPatchSites(workspaceId, scope, true);
  const items = sites.data || [];
  const patchCount = items.reduce((total, item) => total + item.patchCount, 0);

  return <Box sx={{ minHeight: 0, height: "100%", overflow: "hidden", display: "grid", gridTemplateRows: "auto minmax(0, 1fr)" }}>
    <Box sx={{ px: 2, py: 1.25, display: "grid", gap: .75, borderBottom: 1, borderColor: "divider" }}>
      {regionControl}
      <Typography variant="caption" color="text.secondary">{sites.isLoading ? "加载中" : `${items.length} 个训练区域 · ${patchCount} 个 Patch`}</Typography>
    </Box>
    <Box sx={{ minHeight: 0, overflow: "auto" }}>
      {sites.isLoading ? <Box sx={{ p: 3, textAlign: "center" }}><CircularProgress size={24} /></Box>
        : sites.isError ? <Typography color="error" sx={{ p: 2 }}>{sites.error.message}</Typography>
          : !items.length ? <Box sx={{ p: 3, textAlign: "center" }}><Typography variant="body2">当前区域还没有训练数据</Typography></Box>
            : <List disablePadding>
              <ListItemButton divider selected={!selectedSiteId} onClick={() => onSelect("")}>
                <ListItemText primary="全部训练区域" secondary={`${patchCount} 个 Patch`} />
              </ListItemButton>
              {items.map((item) => <ListItemButton key={item.key} divider selected={item.siteId === selectedSiteId} onClick={() => onSelect(item.siteId)} sx={{ alignItems: "flex-start", py: 1.25 }}>
                <ListItemText
                  primary={<Box sx={{ display: "flex", alignItems: "center", gap: .75 }}><Typography variant="body2" noWrap sx={{ flex: 1 }}>{item.displayName}</Typography>{scope === "all" && <Chip size="small" label={item.regionName} />}</Box>}
                  secondary={`${item.patchCount} 个 Patch · 已纳入 ${item.includedCount} 个`}
                />
              </ListItemButton>)}
            </List>}
    </Box>
  </Box>;
}
