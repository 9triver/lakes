import { useMemo, useState } from "react";
import { Alert, Box, Button, Card, CardActionArea, CardContent, Checkbox, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, IconButton, InputLabel, ListItemText, Menu, MenuItem, Select, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography } from "@mui/material";
import { Archive, AlertTriangle, MoreVertical, Pencil, Plus, Waves } from "lucide-react";
import { useArchiveProfile, useCreateProfile, useProfiles, useRenameProfile } from "./api";
import type { TrainingProfile } from "../../api/types";

function ProfileCard({ profile, onSelect, onRename, onArchive }: { profile: TrainingProfile; onSelect: () => void; onRename: () => void; onArchive: () => void }) {
  const [anchor, setAnchor] = useState<null | HTMLElement>(null);
  const menuOpen = Boolean(anchor);
  return (
    <Card variant="outlined" sx={{ height: "100%", position: "relative", borderColor: profile.status === "needs_resolution" ? "warning.main" : "divider" }}>
      <CardActionArea aria-label={`进入用户 ${profile.name}`} onClick={onSelect} sx={{ height: "100%" }}>
        <CardContent sx={{ pr: 6 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: .75, minWidth: 0, mb: 2 }}>
            <Typography variant="subtitle1" color="text.primary" noWrap sx={{ minWidth: 0, flex: 1 }}>{profile.name}</Typography>
            {profile.default && <Typography variant="caption" color="text.secondary">默认</Typography>}
          </Box>
          {profile.status === "needs_resolution" && <Alert icon={<AlertTriangle size={17} />} severity="warning" sx={{ mb: 1, py: 0, alignItems: "center" }}>有 {profile.conflict_count} 个来源冲突</Alert>}
          <Typography variant="body2" color="text.secondary">{profile.site_count} 个观测区域 · {profile.selected_patch_count} 个逻辑 Patch</Typography>
        </CardContent>
      </CardActionArea>
      <Tooltip title="更多操作"><IconButton size="small" aria-label={`${profile.name} 更多操作`} sx={{ position: "absolute", top: 10, right: 10, zIndex: 1, bgcolor: "background.paper" }} onClick={(event) => setAnchor(event.currentTarget)}><MoreVertical size={17} /></IconButton></Tooltip>
      <Menu anchorEl={anchor} open={menuOpen} onClose={() => setAnchor(null)}>
        <MenuItem onClick={() => { setAnchor(null); onRename(); }}><Pencil size={16} style={{ marginRight: 8 }} />重命名</MenuItem>
        {!profile.default && <MenuItem onClick={() => { setAnchor(null); onArchive(); }}><Archive size={16} style={{ marginRight: 8 }} />归档</MenuItem>}
      </Menu>
    </Card>
  );
}

export function ProfileHome({ onSelect }: { onSelect: (profileId: string) => void }) {
  const profiles = useProfiles();
  const create = useCreateProfile();
  const rename = useRenameProfile();
  const archive = useArchiveProfile();
  const [createOpen, setCreateOpen] = useState(false);
  const [mode, setMode] = useState<"empty" | "union">("empty");
  const [name, setName] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [renameTarget, setRenameTarget] = useState<TrainingProfile | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [archiveTarget, setArchiveTarget] = useState<TrainingProfile | null>(null);
  const error = create.error || rename.error || archive.error;
  const activeProfiles = useMemo(() => profiles.data?.items || [], [profiles.data?.items]);
  const navigateToProfile = (profileId: string) => onSelect(profileId);
  const submitCreate = () => create.mutate({ name: name.trim(), mode, source_profile_ids: mode === "union" ? sources : [] }, {
    onSuccess: ({ profile }) => {
      setCreateOpen(false);
      setName("");
      setSources([]);
      navigateToProfile(profile.id);
    },
  });
  const submitRename = () => {
    if (!renameTarget) return;
    rename.mutate({ id: renameTarget.id, name: renameValue.trim() }, { onSuccess: () => setRenameTarget(null) });
  };
  const submitArchive = () => {
    if (!archiveTarget) return;
    archive.mutate({ id: archiveTarget.id, restore: false }, { onSuccess: () => setArchiveTarget(null) });
  };
  if (profiles.isLoading) return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center" }}><CircularProgress /></Box>;
  if (profiles.isError) return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center", p: 3 }}><Alert severity="error">{profiles.error.message}</Alert></Box>;
  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "background.default", px: { xs: 2, sm: 4 }, py: { xs: 3, md: 6 } }}>
      <Box sx={{ width: "min(1120px, 100%)", mx: "auto" }}>
        <Box sx={{ display: "flex", alignItems: { xs: "start", sm: "center" }, justifyContent: "space-between", gap: 2, mb: 4, flexWrap: "wrap" }}>
          <Box><Box sx={{ display: "flex", alignItems: "center", gap: 1 }}><Waves size={23} color="#196b55" /><Typography variant="h5" color="text.primary">Lakes Workbench</Typography></Box><Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>选择一个用户工作空间</Typography></Box>
          <Button variant="contained" startIcon={<Plus size={17} />} onClick={() => setCreateOpen(true)}>新建用户</Button>
        </Box>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error.message}</Alert>}
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", lg: "repeat(3, minmax(0, 1fr))" }, gap: 2 }}>
          {activeProfiles.map((profile) => <ProfileCard key={profile.id} profile={profile} onSelect={() => navigateToProfile(profile.id)} onRename={() => { setRenameTarget(profile); setRenameValue(profile.name); }} onArchive={() => setArchiveTarget(profile)} />)}
        </Box>
        {!activeProfiles.length && <Box sx={{ py: 8, textAlign: "center" }}><Typography color="text.secondary">暂无可用用户</Typography></Box>}
      </Box>
      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>新建用户</DialogTitle>
        <DialogContent sx={{ display: "grid", gap: 2, pt: "8px !important" }}>
          <ToggleButtonGroup exclusive size="small" value={mode} onChange={(_, value) => value && setMode(value)}><ToggleButton value="empty">空白</ToggleButton><ToggleButton value="union">并集</ToggleButton></ToggleButtonGroup>
          <TextField autoFocus label="新用户名称" value={name} onChange={(event) => setName(event.target.value)} />
          {mode === "union" && <FormControl><InputLabel>来源用户</InputLabel><Select multiple label="来源用户" value={sources} onChange={(event) => setSources(typeof event.target.value === "string" ? event.target.value.split(",") : event.target.value)} renderValue={(selected) => selected.map((id) => activeProfiles.find((item) => item.id === id)?.name || id).join(" + ")}>{activeProfiles.filter((item) => item.status === "active").map((item) => <MenuItem key={item.id} value={item.id}><Checkbox checked={sources.includes(item.id)} /><ListItemText primary={item.name} secondary={`${item.selected_patch_count} Patch`} /></MenuItem>)}</Select></FormControl>}
          {mode === "union" && !activeProfiles.some((item) => item.status === "active") && <Alert severity="info">没有可用的来源用户</Alert>}
          {create.error && <Alert severity="error">{create.error.message}</Alert>}
        </DialogContent>
        <DialogActions><Button onClick={() => setCreateOpen(false)}>取消</Button><Button variant="contained" disabled={!name.trim() || (mode === "union" && !sources.length) || create.isPending} onClick={submitCreate}>创建并进入</Button></DialogActions>
      </Dialog>
      <Dialog open={Boolean(renameTarget)} onClose={() => setRenameTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>重命名用户</DialogTitle><DialogContent sx={{ display: "grid", gap: 1.5 }}><TextField autoFocus fullWidth label="用户名称" value={renameValue} onChange={(event) => setRenameValue(event.target.value)} sx={{ mt: 1 }} />{rename.error && <Alert severity="error">{rename.error.message}</Alert>}</DialogContent><DialogActions><Button onClick={() => setRenameTarget(null)}>取消</Button><Button variant="contained" disabled={!renameValue.trim() || rename.isPending} onClick={submitRename}>保存</Button></DialogActions>
      </Dialog>
      <Dialog open={Boolean(archiveTarget)} onClose={() => setArchiveTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>归档用户？</DialogTitle><DialogContent sx={{ display: "grid", gap: 1.5 }}><Typography>归档后该用户不会再出现在选择页，训练数据和模型文件不会删除。</Typography>{archive.error && <Alert severity="error">{archive.error.message}</Alert>}</DialogContent><DialogActions><Button onClick={() => setArchiveTarget(null)}>取消</Button><Button color="warning" variant="contained" disabled={archive.isPending} onClick={submitArchive}>确认归档</Button></DialogActions>
      </Dialog>
    </Box>
  );
}

export function ProfileRouteError({ reason, onBack }: { reason: "missing" | "archived" | "route"; onBack: () => void }) {
  const message = reason === "archived" ? "该用户已归档，无法进入工作区" : reason === "missing" ? "找不到这个用户" : "页面地址无效";
  return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center", p: 3 }}><Box sx={{ textAlign: "center" }}><Typography variant="h5" color="text.primary" gutterBottom>{message}</Typography><Typography color="text.secondary" sx={{ mb: 2 }}>请从用户选择页进入一个可用的工作空间。</Typography><Button variant="contained" onClick={onBack}>返回用户选择</Button></Box></Box>;
}
