import { useEffect, useState } from "react";

/** Temporary display state only: do not overwrite the user's chosen layers. */
export function useControlPeek(enabled: boolean) {
  const [held, setHeld] = useState(false);
  useEffect(() => {
    if (!enabled) { setHeld(false); return; }
    const down = (event: KeyboardEvent) => {
      if (event.key === "Control" || event.ctrlKey) setHeld(true);
    };
    const up = (event: KeyboardEvent) => {
      if (event.key === "Control" || !event.ctrlKey) setHeld(event.ctrlKey);
    };
    const reset = () => setHeld(false);
    const visibility = () => { if (document.hidden) reset(); };
    window.addEventListener("keydown", down, true);
    window.addEventListener("keyup", up, true);
    window.addEventListener("blur", reset);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("keydown", down, true);
      window.removeEventListener("keyup", up, true);
      window.removeEventListener("blur", reset);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [enabled]);
  return enabled && held;
}
