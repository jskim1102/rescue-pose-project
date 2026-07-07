import assert from "node:assert/strict";

import {
  CRITICAL_LYING_SEC,
  FALL_CONFIRM_MS,
  RECOVERY_CONFIRM_MS,
  type RescueIncidentTimelineState,
  riskForPosture,
  updateRescueIncidentTimeline,
} from "../src/pages/rescueIncidentTimeline";

function newState(): RescueIncidentTimelineState {
  return {};
}

function standing(nowMs: number) {
  return {
    label: "CAM 01",
    people: 1,
    sitting: 0,
    lying: 0,
    rescueNeeded: 0,
    maxLyingSec: 0,
    patientId: "P-01",
    nowMs,
  };
}

function sitting(nowMs: number) {
  return {
    label: "CAM 01",
    people: 1,
    sitting: 1,
    lying: 0,
    rescueNeeded: 0,
    maxLyingSec: 0,
    patientId: "P-01",
    nowMs,
  };
}

function lying(nowMs: number, lyingSec = 0) {
  return {
    label: "CAM 01",
    people: 1,
    sitting: 0,
    lying: 1,
    rescueNeeded: 1,
    maxLyingSec: lyingSec,
    patientId: "P-01",
    nowMs,
  };
}

function empty(nowMs: number) {
  return {
    label: "CAM 01",
    people: 0,
    sitting: 0,
    lying: 0,
    rescueNeeded: 0,
    maxLyingSec: 0,
    patientId: "P-01",
    nowMs,
  };
}

{
  assert.equal(riskForPosture("standing", 0), "LOW");
  assert.equal(riskForPosture("sitting", 0), "MEDIUM");
  assert.equal(riskForPosture("lying", CRITICAL_LYING_SEC - 1), "HIGH");
  assert.equal(riskForPosture("lying", CRITICAL_LYING_SEC), "CRITICAL");
}

{
  const state = newState();

  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(1_000)], 1_000), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [empty(1_300)], 1_300), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(1_600)], 1_600), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(3_000)], 3_000), []);
}

{
  const state = newState();

  assert.deepEqual(updateRescueIncidentTimeline(state, [lying(1_000)], 1_000), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [lying(1_000 + FALL_CONFIRM_MS - 1)], 1_000 + FALL_CONFIRM_MS - 1), []);

  const started = updateRescueIncidentTimeline(
    state,
    [lying(1_000 + FALL_CONFIRM_MS, 1)],
    1_000 + FALL_CONFIRM_MS,
  );
  assert.equal(started.length, 1);
  assert.equal(started[0].kind, "위험 자세 감지 (Risk Posture)");
  assert.equal(started[0].patientId, "P-01");
  assert.equal(started[0].risk, "HIGH");
  assert.equal(started[0].detail, "누운 자세 감지 (1초 경과)");

  assert.deepEqual(updateRescueIncidentTimeline(state, [lying(3_000, 2)], 3_000), []);
}

{
  const state = newState();

  assert.deepEqual(updateRescueIncidentTimeline(state, [sitting(1_000)], 1_000), []);
  const started = updateRescueIncidentTimeline(
    state,
    [sitting(1_000 + FALL_CONFIRM_MS)],
    1_000 + FALL_CONFIRM_MS,
  );
  assert.equal(started.length, 1);
  assert.equal(started[0].kind, "위험 자세 감지 (Risk Posture)");
  assert.equal(started[0].risk, "MEDIUM");
  assert.equal(started[0].detail, "앉은 자세 감지");
  assert.deepEqual(updateRescueIncidentTimeline(state, [sitting(3_000)], 3_000), []);
}

{
  const state = newState();

  updateRescueIncidentTimeline(state, [lying(1_000)], 1_000);
  updateRescueIncidentTimeline(state, [lying(1_000 + FALL_CONFIRM_MS, 1)], 1_000 + FALL_CONFIRM_MS);
  assert.deepEqual(
    updateRescueIncidentTimeline(state, [lying(8_000, CRITICAL_LYING_SEC - 1)], 8_000),
    [],
  );

  const escalated = updateRescueIncidentTimeline(state, [lying(9_000, CRITICAL_LYING_SEC)], 9_000);
  assert.equal(escalated.length, 1);
  assert.equal(escalated[0].kind, "위험도 상승 (Risk Escalated)");
  assert.equal(escalated[0].risk, "CRITICAL");
  assert.equal(escalated[0].detail, "누운 자세 10초 이상 지속 (10초 경과)");
  assert.deepEqual(updateRescueIncidentTimeline(state, [lying(10_000, 11)], 10_000), []);
}

{
  const state = newState();

  updateRescueIncidentTimeline(state, [lying(1_000)], 1_000);
  updateRescueIncidentTimeline(state, [lying(1_000 + FALL_CONFIRM_MS, 1)], 1_000 + FALL_CONFIRM_MS);

  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(2_000)], 2_000), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [lying(2_300, 1)], 2_300), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(3_000)], 3_000), []);

  const recovered = updateRescueIncidentTimeline(
    state,
    [standing(3_000 + RECOVERY_CONFIRM_MS)],
    3_000 + RECOVERY_CONFIRM_MS,
  );
  assert.equal(recovered.length, 1);
  assert.equal(recovered[0].kind, "구조 상황 해제 (Recovered)");
  assert.equal(recovered[0].patientId, "P-01");

  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(5_000)], 5_000), []);
}

{
  const state = newState();

  updateRescueIncidentTimeline(state, [lying(1_000)], 1_000);
  updateRescueIncidentTimeline(state, [lying(1_000 + FALL_CONFIRM_MS, 1)], 1_000 + FALL_CONFIRM_MS);

  assert.deepEqual(updateRescueIncidentTimeline(state, [empty(3_000)], 3_000), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [empty(5_000)], 5_000), []);
  assert.deepEqual(updateRescueIncidentTimeline(state, [standing(6_000)], 6_000), []);

  const recovered = updateRescueIncidentTimeline(
    state,
    [standing(6_000 + RECOVERY_CONFIRM_MS)],
    6_000 + RECOVERY_CONFIRM_MS,
  );
  assert.equal(recovered.length, 1);
  assert.equal(recovered[0].kind, "구조 상황 해제 (Recovered)");
}
