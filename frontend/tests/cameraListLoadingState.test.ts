import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const appSource = readFileSync(resolve(here, "../src/App.tsx"), "utf8");
const settingsSource = readFileSync(resolve(here, "../src/pages/SettingsPage.tsx"), "utf8");
const dashboardSource = readFileSync(resolve(here, "../src/pages/DashboardPage.tsx"), "utf8");
const registryPath = resolve(here, "../src/hooks/useCameraRegistry.tsx");

assert.ok(
  existsSync(registryPath),
  "Camera registry state should live above routed pages so navigation can reuse it immediately",
);
const registrySource = readFileSync(registryPath, "utf8");

assert.ok(
  registrySource.includes('useState<CameraListStatus>("loading")') &&
    registrySource.includes("setCams(data as RegisteredCamera[])"),
  "The shared registry should own the camera list and its loading state",
);
assert.ok(
  appSource.includes("<CameraRegistryProvider>") &&
    appSource.indexOf("<CameraRegistryProvider>") < appSource.indexOf("<Routes>"),
  "The registry provider should stay mounted above route transitions",
);
assert.ok(
  registrySource.includes("useLocation()") &&
    registrySource.includes("[refreshCameras, pathname]"),
  "Route transitions should revalidate the cached list without resetting it to loading",
);
assert.ok(
  settingsSource.includes("useCameraRegistry()") &&
    !settingsSource.includes("useState<IpCam[]>([])"),
  "Settings should reuse the shared camera list instead of starting from an empty local list",
);
assert.ok(
  dashboardSource.includes("useCameraRegistry()") &&
    !dashboardSource.includes("useState<IpCam[]>([])"),
  "Dashboard should publish its loaded cameras to the same registry used by Settings",
);
assert.ok(
  settingsSource.includes('camsStatus === "ready" ? cams.length : "—"'),
  "Settings should not display a zero count before the camera list is confirmed",
);
assert.ok(
  settingsSource.includes("카메라 목록 불러오는 중…") &&
    dashboardSource.includes("카메라 목록 불러오는 중…"),
  "Both pages should distinguish the initial load from a confirmed empty registry",
);
assert.ok(
  settingsSource.includes("카메라 목록을 불러오지 못했습니다") &&
    dashboardSource.includes("카메라 목록을 불러오지 못했습니다"),
  "Both pages should distinguish an API failure from a confirmed empty registry",
);
assert.ok(
  registrySource.includes('status === "ready" ? status : "error"'),
  "A transient refresh failure must preserve the last successful camera list",
);
