import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const KEY = "theme";

/** The palette lives entirely in CSS variables keyed off [data-theme] on <html>, so the
 *  only job here is writing that attribute and remembering the choice. Light is the
 *  default; a workbench that gets read next to a screen in daylight starts light. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, set] = useState<Theme>(
    () => (localStorage.getItem(KEY) as Theme) || "light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(KEY, theme);
  }, [theme]);

  return [theme, set];
}
