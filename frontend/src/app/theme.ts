import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#176b55", dark: "#10513f", light: "#d7e8e1" },
    background: { default: "#e9ecee", paper: "#f7f8f5" },
    divider: "#c8d0d2",
    text: { primary: "#1e2925", secondary: "#5c6863" },
  },
  shape: { borderRadius: 4 },
  typography: { fontFamily: "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { backgroundColor: "#e9ecee" },
      },
    },
    MuiButton: { defaultProps: { size: "small", disableElevation: true } },
    MuiFormControl: { defaultProps: { size: "small" } },
    MuiTextField: { defaultProps: { size: "small" } },
    MuiOutlinedInput: {
      styleOverrides: {
        root: { backgroundColor: "#fcfdfb" },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          "&.Mui-selected": { backgroundColor: "#d7e5df" },
          "&.Mui-selected:hover": { backgroundColor: "#ccdcd5" },
        },
      },
    },
  },
});
