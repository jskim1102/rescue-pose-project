export const PATIENT_ID_GRACE_MS = 10_000;

export interface PatientEpisode {
  id: string;
  lastSeenMs: number;
}

export type PatientEpisodeMap = Record<string, PatientEpisode>;

export function getPatientEpisodeId(
  episodes: PatientEpisodeMap,
  label: string,
  nowMs: number,
  graceMs: number = PATIENT_ID_GRACE_MS,
): string | null {
  const episode = episodes[label];
  if (!episode) return null;
  return nowMs - episode.lastSeenMs <= graceMs ? episode.id : null;
}

export function ensurePatientEpisodeId(
  episodes: PatientEpisodeMap,
  label: string,
  nowMs: number,
  nextId: () => string,
  graceMs: number = PATIENT_ID_GRACE_MS,
): string {
  const current = getPatientEpisodeId(episodes, label, nowMs, graceMs);
  if (current) {
    episodes[label].lastSeenMs = nowMs;
    return current;
  }

  const id = nextId();
  episodes[label] = { id, lastSeenMs: nowMs };
  return id;
}
