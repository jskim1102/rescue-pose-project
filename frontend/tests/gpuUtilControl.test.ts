import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  GpuUtilTargetUpdater,
  normalizeGpuUtilConfig,
} from "../src/utils/gpuUtilControl.ts";


const wait = (milliseconds: number) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));
const here = dirname(fileURLToPath(import.meta.url));

const configResponse = (target: number, duty = 0.5) =>
  new Response(
    JSON.stringify({
      gpu_util_target: target,
      gpu_util_duty: duty,
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }
  );


test("연속 target 변경은 debounce 후 마지막 값만 PUT한다", async () => {
  const requests: Array<{ body: string; signal: AbortSignal }> = [];
  const updater = new GpuUtilTargetUpdater({
    endpoint: "/api/inference/config",
    initialTarget: 1,
    debounceMs: 5,
    fetcher: async (_input, init) => {
      requests.push({
        body: String(init?.body),
        signal: init?.signal as AbortSignal,
      });
      return configResponse(0.6);
    },
  });

  updater.schedule(0.5);
  updater.schedule(0.7);
  updater.schedule(0.6);
  await wait(25);

  assert.equal(requests.length, 1);
  assert.deepEqual(JSON.parse(requests[0].body), { gpu_util_target: 0.6 });
  updater.dispose();
});


test("새 target은 이전 in-flight PUT을 abort하고 늦은 응답을 무시한다", async () => {
  let resolveFirst: ((response: Response) => void) | undefined;
  const firstResponse = new Promise<Response>((resolve) => {
    resolveFirst = resolve;
  });
  const signals: AbortSignal[] = [];
  const accepted: number[] = [];
  let requestIndex = 0;
  const updater = new GpuUtilTargetUpdater({
    endpoint: "/api/inference/config",
    initialTarget: 1,
    debounceMs: 5,
    fetcher: async (_input, init) => {
      signals.push(init?.signal as AbortSignal);
      requestIndex += 1;
      return requestIndex === 1 ? firstResponse : configResponse(0.7);
    },
    onAccepted: (config) => accepted.push(config.gpu_util_target),
  });

  updater.schedule(0.5);
  await wait(15);
  updater.schedule(0.7);
  assert.equal(signals[0].aborted, true);
  await wait(15);
  resolveFirst?.(configResponse(0.5));
  await wait(0);

  assert.deepEqual(accepted, [0.7]);
  updater.dispose();
});


test("PUT 실패 시 마지막 서버 target으로 복원한다", async () => {
  const restored: number[] = [];
  const updater = new GpuUtilTargetUpdater({
    endpoint: "/api/inference/config",
    initialTarget: 0.8,
    debounceMs: 5,
    fetcher: async () => new Response(null, { status: 503 }),
    onRejected: (lastTarget) => restored.push(lastTarget),
  });

  updater.schedule(0.4);
  await wait(25);

  assert.deepEqual(restored, [0.8]);
  updater.dispose();
});


test("GPU target은 UI에서도 95%를 넘지 않는다", () => {
  assert.equal(normalizeGpuUtilConfig({ gpu_util_target: 1 }).gpu_util_target, 0.95);
  assert.equal(normalizeGpuUtilConfig({ gpu_util_target: 0.01 }).gpu_util_target, 0.1);
});


test("실제 설정 화면 slider 상한도 95%다", () => {
  const settingsSource = readFileSync(
    resolve(here, "../src/pages/SettingsPage.tsx"),
    "utf8",
  );
  assert.match(settingsSource, /DEFAULT_GPU_UTIL_TARGET_PCT = 95/);
  assert.match(settingsSource, /id="gpu-util-target"[\s\S]*?max="95"/);
});
