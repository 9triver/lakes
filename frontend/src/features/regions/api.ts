import { useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { RegionsResponse } from "../../api/types";

export function useRegions() {
  return useQuery({ queryKey: ["regions"], queryFn: () => getJson<RegionsResponse>("/api/regions") });
}
