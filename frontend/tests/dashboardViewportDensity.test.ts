import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dashboardSource = readFileSync(resolve(here, "../src/pages/DashboardPage.tsx"), "utf8");
const topbarSource = readFileSync(resolve(here, "../src/components/Topbar.tsx"), "utf8");
const globalCss = readFileSync(resolve(here, "../src/styles.css"), "utf8");

assert.ok(
  globalCss.includes("--rp-cam-body-max-h"),
  "global CSS should define a viewport-aware camera body height token",
);
assert.ok(
  globalCss.includes("font-size: clamp(13px, 0.74vw, 16px)"),
  "root font size should compact on FHD and cap on larger monitors",
);
assert.ok(
  dashboardSource.includes('minHeight: "100vh"') && !dashboardSource.includes('height: "100vh"'),
  "Dashboard root should not force the content grid to stretch to the viewport height",
);
assert.ok(
  dashboardSource.includes('alignItems: "stretch"') && !dashboardSource.includes('body: { flex: 1'),
  "Dashboard body should stretch columns to content height, not viewport height",
);
assert.ok(
  dashboardSource.includes('maxHeight: "var(--rp-cam-body-max-h)"'),
  "camera bodies should have a viewport-aware max height",
);
assert.ok(
  dashboardSource.includes('height: "var(--rp-event-row-h)"'),
  "event rows should use compact viewport-aware heights",
);
assert.ok(
  topbarSource.includes('padding: "var(--rp-topbar-pad-y) var(--rp-topbar-pad-x)"'),
  "Topbar should use the shared density tokens",
);
