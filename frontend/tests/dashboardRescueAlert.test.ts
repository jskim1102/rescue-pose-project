import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dashboardSource = readFileSync(resolve(here, "../src/pages/DashboardPage.tsx"), "utf8");

assert.ok(
  !dashboardSource.includes('from "../components/RescueAlert"'),
  "DashboardPage should not import the top rescue alert banner",
);
assert.ok(
  !dashboardSource.includes("<RescueAlert"),
  "DashboardPage should not render the top rescue alert banner",
);
