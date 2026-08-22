import { useQuery } from "@tanstack/react-query";
import { getJson } from "../../api/client";
import type { AuthSession } from "../../api/types";

export function useAuthSession() {
  return useQuery({
    queryKey: ["auth-session"],
    queryFn: () => getJson<AuthSession>("/api/auth/session"),
    retry: false,
    staleTime: 60_000,
  });
}
