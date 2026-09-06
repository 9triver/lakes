import { Box, Button, Typography } from "@mui/material";
import { LogOut } from "lucide-react";
import type { WorkbenchUser } from "../../api/types";

export function CurrentUserBar({ user, logoutUrl, compact = false }: { user?: WorkbenchUser; logoutUrl?: string; compact?: boolean }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 0, py: compact ? .5 : 0 }}>
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="caption" color="text.secondary" display="block">当前用户</Typography>
        <Typography variant="body2" color="text.primary" noWrap>{user?.name || "加载中"}</Typography>
        <Typography variant="caption" color="text.secondary" display="block" noWrap>{user?.workspace.name || "训练工作区"}</Typography>
      </Box>
      {logoutUrl && <Button size="small" variant="text" startIcon={<LogOut size={15} />} href={logoutUrl}>退出</Button>}
    </Box>
  );
}
