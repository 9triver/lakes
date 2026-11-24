import { Box, Button, Menu, MenuItem, Tab, Tabs, Typography, ToggleButton, ToggleButtonGroup } from "@mui/material";
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
  const [validationMode,setValidationMode]=useState('single');
  const [galleryMenu, setGalleryMenu] = useState<HTMLElement | null>(null);
  const [comparisonOpened, setComparisonOpened] = useState(false);
  const comparisonProps = useRef(validationProps);
  if (validationMode === "compare") comparisonProps.current = validationProps;
  const switchValidationMode = (value: string | null) => {
    if (!value) return;
    if (value === "compare") {
      comparisonProps.current = validationProps;
      setComparisonOpened(true);
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
      {stage === "validate" && <ToggleButtonGroup size="small" exclusive value={validationMode} onChange={(_,value)=>switchValidationMode(value)}><ToggleButton value="single">单模型地图</ToggleButton><ToggleButton value="compare">多模型对比</ToggleButton></ToggleButtonGroup>}
      {stage === "validate" && validationMode === "single" && <>
        <Button
          size="small"
          aria-controls={galleryMenu ? "sample-gallery-menu" : undefined}
          aria-haspopup="menu"
          aria-expanded={Boolean(galleryMenu)}
          onClick={(event) => setGalleryMenu(event.currentTarget)}
        >样本资料</Button>
        <Menu id="sample-gallery-menu" anchorEl={galleryMenu} open={Boolean(galleryMenu)} onClose={() => setGalleryMenu(null)}>
          <MenuItem component="a" href="/assets/sample-galleries/eleven_provinces_cluster_gallery.html" target="_blank" rel="noopener noreferrer" onClick={() => setGalleryMenu(null)}>
            16 省级地区训练样本聚类
          </MenuItem>
          <MenuItem component="a" href="/assets/sample-galleries/eleven_provinces_visual_taxonomy.html" target="_blank" rel="noopener noreferrer" onClick={() => setGalleryMenu(null)}>
            14 省水体与困难非水体分类
          </MenuItem>
        </Menu>
      </>}
    </Box>
    <Box sx={{ minHeight: 0, overflow: "hidden" }}>
      {stage === "train" ? <TrainingView workspaceId={workspaceId} scope={scope} regionOptions={regionOptions} onScopeChange={onScopeChange} defaults={defaults} selectedRunId={selectedRunId} onRunSelect={onRunSelect} /> : <>
        {comparisonOpened && <Box sx={{ height: "100%", display: validationMode === "compare" ? "block" : "none" }}>
          <ModelComparisonView workspaceId={workspaceId} scope={scope} {...comparisonProps.current} />
        </Box>}
        {validationMode === "single" && <ModelValidationView workspaceId={workspaceId} scope={scope} onTrainingDataGenerated={onTrainingDataGenerated} {...validationProps} />}
      </>}
    </Box>
  </Box>;
}
