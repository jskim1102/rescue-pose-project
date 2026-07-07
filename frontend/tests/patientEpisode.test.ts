import assert from "node:assert/strict";

import {
  PATIENT_ID_GRACE_MS,
  type PatientEpisodeMap,
  ensurePatientEpisodeId,
  getPatientEpisodeId,
} from "../src/pages/patientEpisode";

function nextIdFactory() {
  let seq = 0;
  return () => `P-${String(++seq).padStart(2, "0")}`;
}

{
  const episodes: PatientEpisodeMap = {};
  const nextId = nextIdFactory();

  assert.equal(ensurePatientEpisodeId(episodes, "CAM 02", 1_000, nextId), "P-01");
  assert.equal(getPatientEpisodeId(episodes, "CAM 02", 1_000 + PATIENT_ID_GRACE_MS - 1), "P-01");
  assert.equal(getPatientEpisodeId(episodes, "CAM 02", 1_000 + PATIENT_ID_GRACE_MS + 1), null);
}

{
  const episodes: PatientEpisodeMap = {};
  const nextId = nextIdFactory();

  assert.equal(ensurePatientEpisodeId(episodes, "CAM 02", 1_000, nextId), "P-01");
  assert.equal(ensurePatientEpisodeId(episodes, "CAM 02", 6_000, nextId), "P-01");
  assert.equal(ensurePatientEpisodeId(episodes, "CAM 02", 16_001, nextId), "P-02");
}

{
  const episodes: PatientEpisodeMap = {};
  const nextId = nextIdFactory();

  assert.equal(ensurePatientEpisodeId(episodes, "CAM 01", 1_000, nextId), "P-01");
  assert.equal(ensurePatientEpisodeId(episodes, "CAM 02", 1_000, nextId), "P-02");
  assert.equal(ensurePatientEpisodeId(episodes, "CAM 01", 2_000, nextId), "P-01");
}
