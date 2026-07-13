import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import type { SiteFilters } from "../../api/types";

const options: Array<{ key: keyof SiteFilters; label: string; items: Array<[string, string]> }> = [
  { key: "area_bucket", label: "覆盖面积", items: [["", "全部面积"], ["gte100", "≥100 km²"], ["10_100", "10-100 km²"], ["1_10", "1-10 km²"], ["0_1_1", "0.1-1 km²"], ["lt0_1", "<0.1 km²"]] },
  { key: "has_name", label: "名称提示", items: [["", "全部"], ["true", "有提示"], ["false", "无提示"]] },
  { key: "has_tci", label: "影像", items: [["", "全部影像"], ["true", "有影像"], ["false", "无影像"]] },
  { key: "has_osm", label: "OSM 标注", items: [["", "全部"], ["true", "有候选"], ["false", "无候选"]] },
  { key: "has_hydrolakes", label: "HydroLAKES", items: [["", "全部"], ["true", "有候选"], ["false", "无候选"]] },
  { key: "has_local_labels", label: "本地标注", items: [["", "全部"], ["true", "有标注"], ["false", "无标注"]] },
];

export function SiteFilterControls({ value, onChange }: { value: SiteFilters; onChange: (filters: SiteFilters) => void }) {
  return <>{options.map((option) => <FormControl key={option.key} fullWidth>
    <InputLabel id={`${option.key}-filter-label`}>{option.label}</InputLabel>
    <Select labelId={`${option.key}-filter-label`} label={option.label} value={value[option.key]} onChange={(event) => onChange({ ...value, [option.key]: event.target.value })}>
      {option.items.map(([itemValue, label]) => <MenuItem key={itemValue} value={itemValue}>{label}</MenuItem>)}
    </Select>
  </FormControl>)}</>;
}
