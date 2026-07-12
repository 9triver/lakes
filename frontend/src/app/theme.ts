import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#196b55" },
    background: { default: "#f7f8f5", paper: "#ffffff" },
  },
  shape: { borderRadius: 4 },
  typography: { fontFamily: "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" },
  components: {
    MuiButton: { defaultProps: { size: "small", disableElevation: true } },
    MuiFormControl: { defaultProps: { size: "small" } },
    MuiTextField: { defaultProps: { size: "small" } },
  },
});
