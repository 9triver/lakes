import { Box, Tab, Tabs, Typography } from "@mui/material";
import { ModelValidationView } from "../model-validation/ModelValidationView";
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
  return <Box sx={{ height: "100%", minHeight: 0, display: "grid", gridTemplateRows: "auto minmax(0,1fr)", overflow: "hidden" }}>
    <Box sx={{ px: { xs: 1.5, sm: 2 }, bgcolor: "background.paper", borderBottom: 1, borderColor: "divider", display: "flex", alignItems: "center", gap: 2 }}>
      <Typography variant="h6" color="text.primary" sx={{ py: 1.25 }}>模型实验</Typography>
      <Tabs value={stage} onChange={(_, value: ModelStage) => onStageChange(value)} aria-label="模型工作台">
        <Tab value="train" label="训练" />
        <Tab value="validate" label="验证" />
      </Tabs>
    </Box>
    <Box sx={{ minHeight: 0, overflow: "hidden" }}>
      {stage === "train" ? <TrainingView workspaceId={workspaceId} scope={scope} regionOptions={regionOptions} onScopeChange={onScopeChange} defaults={defaults} selectedRunId={selectedRunId} onRunSelect={onRunSelect} /> : <ModelValidationView workspaceId={workspaceId} scope={scope} onTrainingDataGenerated={onTrainingDataGenerated} {...validationProps} />}
    </Box>
  </Box>;
}
