import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import type { LakeFilters } from "../../api/types";

const options: Array<{ key: keyof LakeFilters; label: string; items: Array<[string, string]> }> = [
  { key: "water_type", label: "类型", items: [["", "全部类型"], ["lake", "湖泊"], ["reservoir", "水库"], ["pond", "坑塘"], ["pond_candidate", "疑似坑塘"], ["wetland", "湿地"], ["aquaculture", "养殖水面"], ["unknown", "未分类"]] },
  { key: "area_bucket", label: "面积", items: [["", "全部面积"], ["gte100", "≥100 km²"], ["10_100", "10-100 km²"], ["1_10", "1-10 km²"], ["0_1_1", "0.1-1 km²"], ["lt0_1", "<0.1 km²"]] },
  { key: "has_name", label: "名称", items: [["", "全部名称"], ["true", "有名称"], ["false", "无名称"]] },
  { key: "has_tci", label: "影像", items: [["", "全部影像"], ["true", "有影像"], ["false", "无影像"]] },
  { key: "polygon_quality", label: "边界", items: [["", "全部边界"], ["high", "High"], ["medium", "Medium"], ["low", "Low"]] },
  { key: "metadata_quality", label: "元数据", items: [["", "全部元数据"], ["high", "High"], ["medium", "Medium"], ["low", "Low"]] },
];

export function LakeFilterControls({ value, onChange }: { value: LakeFilters; onChange: (filters: LakeFilters) => void }) {
  return <>{options.map((option) => <FormControl key={option.key} fullWidth>
    <InputLabel>{option.label}</InputLabel>
    <Select label={option.label} value={value[option.key]} onChange={(event) => onChange({ ...value, [option.key]: event.target.value })}>
      {option.items.map(([itemValue, label]) => <MenuItem key={itemValue} value={itemValue}>{label}</MenuItem>)}
    </Select>
  </FormControl>)}</>;
}
