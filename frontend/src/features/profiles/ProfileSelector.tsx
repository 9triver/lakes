import { Box, Button, Chip, Typography } from "@mui/material";
import { ArrowLeftRight } from "lucide-react";
import type { TrainingProfile } from "../../api/types";

export function CurrentProfileBar({ profile, onSwitch, compact = false }: { profile?: TrainingProfile; onSwitch: () => void; compact?: boolean }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 0, py: compact ? .5 : 0 }}>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="caption" color="text.secondary" display="block">当前用户</Typography>
        <Typography variant="body2" color="text.primary" noWrap>{profile?.name || "加载中"}</Typography>
      </Box>
      {profile?.status === "needs_resolution" && <Chip size="small" color="warning" label={`冲突 ${profile.conflict_count}`} />}
      <Button size="small" variant="text" startIcon={<ArrowLeftRight size={15} />} onClick={onSwitch}>切换用户</Button>
    </Box>
  );
}
