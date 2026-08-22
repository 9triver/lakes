import { Box, Button, Typography } from "@mui/material";

export function UserRouteError({ reason, onBack }: { reason: "missing" | "archived" | "route"; onBack: () => void }) {
  const message = reason === "archived" ? "该用户已归档" : reason === "missing" ? "找不到这个用户或训练工作区" : "页面地址无效";
  return <Box sx={{ minHeight: "100vh", display: "grid", placeItems: "center", p: 3 }}><Box sx={{ textAlign: "center" }}><Typography variant="h5" color="text.primary" gutterBottom>{message}</Typography><Button variant="contained" onClick={onBack}>返回当前工作区</Button></Box></Box>;
}
