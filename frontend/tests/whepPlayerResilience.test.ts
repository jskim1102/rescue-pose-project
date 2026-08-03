import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const playerSource = readFileSync(
  new URL("../src/components/WhepPlayer.tsx", import.meta.url),
  "utf8",
);
const apiSource = readFileSync(
  new URL("../src/hooks/useApi.ts", import.meta.url),
  "utf8",
);

test("WHEP reconnects with bounded exponential backoff", () => {
  assert.match(playerSource, /const MAX_RETRIES = 5/);
  assert.match(playerSource, /Math\.min\(1000 \* 2 \*\* attempt, 15000\)/);
  assert.match(playerSource, /state === "failed" \|\| state === "disconnected"/);
  assert.match(playerSource, /window\.setTimeout\(connect, delay\)/);
});

test("WHEP and API ports must come from the offset-backed Vite env", () => {
  assert.match(apiSource, /VITE_API_PORT is required/);
  assert.match(apiSource, /VITE_MEDIAMTX_WEBRTC_PORT is required/);
  assert.doesNotMatch(apiSource, /VITE_API_PORT \|\|/);
  assert.doesNotMatch(apiSource, /VITE_MEDIAMTX_WEBRTC_PORT \|\|/);
});
