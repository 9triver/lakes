import { Box, Tab, Tabs, Typography, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { useRef, useState } from "react";
import { ModelValidationView } from "../model-validation/ModelValidationView";
import { ModelComparisonView } from "../model-validation/ModelComparisonView";
import { TrainingView } from "../training/TrainingView";
import type { RegionSummary } from "../../api/types";

export type ModelStage = "train" | "validate";

interface ModelWorkbenchProps {
  workspaceId: string;
  scope: string;
  regionOptions?: RegionSummary[];
  onScopeChange: (scope: string) => void;
  stage: ModelStage;
  defaults?: Record<string, unknown>;
  routeSiteId?: string;
  routeSiteRegion?: string;
  routeModel?: string;
  allowForeignModels?: boolean;
  onStageChange: (stage: ModelStage) => void;
  onBack: () => void;
  onRouteResult: (siteId: string, siteRegion: string, model: string) => void;
  onRouteModel: (model: string) => void;
  onTrainingDataGenerated: (region: string, sampleId: string, siteId: string) => void;
  selectedRunId?: string;
  onRunSelect?: (runId: string) => void;
}

export function ModelWorkbench({ workspaceId, scope, regionOptions = [], onScopeChange, stage, defaults = {}, selectedRunId = "", onRunSelect, onStageChange, onTrainingDataGenerated, ...validationProps }: ModelWorkbenchProps) {
  const [validationMode,setValidationMode]=useState('compare');
  const comparisonProps = useRef(validationProps);
  if (validationMode === "compare") comparisonProps.current = validationProps;
  const switchValidationMode = (value: string | null) => {
    if (!value) return;
    if (value === "compare" && validationMode !== "compare") {
      const previous = comparisonProps.current;
      previous.onRouteResult(previous.routeSiteId || "", previous.routeSiteRegion || scope, "");
    }
    setValidationMode(value);
  };
  return <Box sx={{ height: "100%", minHeight: 0, display: "grid", gridTemplateRows: "auto minmax(0,1fr)", overflow: "hidden" }}>
    <Box sx={{ px: { xs: 1.5, sm: 2 }, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 2 }}>
      <Typography variant="h6" color="text.primary" sx={{ py: 1.25 }}>模型实验</Typography>
      <Tabs value={stage} onChange={(_, value: ModelStage) => onStageChange(value)} aria-label="模型工作台">
        <Tab value="train" label="训练" />
        <Tab value="validate" label="验证" />
      </Tabs>
      {stage === "validate" && <ToggleButtonGroup size="small" exclusive value={validationMode} onChange={(_,value)=>switchValidationMode(value)}><ToggleButton value="compare">多模型对比</ToggleButton><ToggleButton value="single">单模型地图</ToggleButton></ToggleButtonGroup>}
    </Box>
    <Box sx={{ minHeight: 0, overflow: "hidden" }}>
      {stage === "train" ? <TrainingView workspaceId={workspaceId} scope={scope} regionOptions={regionOptions} onScopeChange={onScopeChange} defaults={defaults} selectedRunId={selectedRunId} onRunSelect={onRunSelect} /> : <><Box sx={{ height: "100%", display: validationMode === "compare" ? "block" : "none" }}><ModelComparisonView workspaceId={workspaceId} scope={scope} {...comparisonProps.current} /></Box>{validationMode === "single" && <ModelValidationView workspaceId={workspaceId} scope={scope} onTrainingDataGenerated={onTrainingDataGenerated} {...validationProps} />}</>}
    </Box>
  </Box>;
}
