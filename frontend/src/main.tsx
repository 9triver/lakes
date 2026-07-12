import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { HashRouter } from "react-router-dom";
import "ol/ol.css";
import { App } from "./app/App";
import { theme } from "./app/theme";

const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <HashRouter><App /></HashRouter>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
