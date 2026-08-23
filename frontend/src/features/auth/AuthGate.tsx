import { Alert, Box, Button, CircularProgress, Typography } from "@mui/material";
import { LogIn, Waves } from "lucide-react";

export function AuthLoading() {
  return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center" }}><Box sx={{ textAlign: "center" }}><CircularProgress size={28} /><Typography color="text.secondary" sx={{ mt: 1.5 }}>正在验证登录状态</Typography></Box></Box>;
}

export function AuthRequired({ message }: { message: string }) {
  return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center", p: 3, bgcolor: "background.default" }}><Box sx={{ width: "min(420px, 100%)", display: "grid", gap: 2, textAlign: "center" }}><Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 1 }}><Waves size={24} color="#196b55" /><Typography variant="h5">Lakes Workbench</Typography></Box><Alert severity="warning">{message}</Alert><Button variant="contained" startIcon={<LogIn size={17} />} onClick={() => window.location.reload()}>登录</Button></Box></Box>;
}
