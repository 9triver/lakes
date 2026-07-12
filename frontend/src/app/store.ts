import { create } from "zustand";

interface WorkbenchState {
  region: string;
  setRegion: (region: string) => void;
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  region: "",
  setRegion: (region) => set({ region }),
}));
