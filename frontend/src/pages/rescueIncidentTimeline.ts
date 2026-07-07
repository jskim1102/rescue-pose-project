export const FALL_CONFIRM_MS = 700;
export const RECOVERY_CONFIRM_MS = 1_000;
export const CRITICAL_LYING_SEC = 10;
const ABSENT_FLICKER_GRACE_MS = 1_000;

export type IncidentRisk = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
export type IncidentPosture = "standing" | "sitting" | "lying";
export type RescueIncidentPhase = "clear" | "risk_candidate" | "active" | "recovering";

export interface RescueIncidentObservation {
  label: string;
  people: number;
  sitting: number;
  lying: number;
  rescueNeeded: number;
  maxLyingSec: number;
  patientId: string;
}

export interface RescueIncidentEvent {
  kind:
    | "위험 자세 감지 (Risk Posture)"
    | "위험도 상승 (Risk Escalated)"
    | "구조 상황 해제 (Recovered)";
  cam: string;
  patientId: string;
  risk: IncidentRisk;
  detail: string;
}

interface CameraIncidentState {
  phase: RescueIncidentPhase;
  patientId: string | null;
  candidateSinceMs: number | null;
  recoverySinceMs: number | null;
  lastSeenMs: number | null;
  candidatePosture: IncidentPosture | null;
  activePosture: IncidentPosture | null;
  criticalLogged: boolean;
}

export type RescueIncidentTimelineState = Record<string, CameraIncidentState>;

function getCameraState(state: RescueIncidentTimelineState, label: string): CameraIncidentState {
  const current = state[label];
  if (current) return current;

  const next: CameraIncidentState = {
    phase: "clear",
    patientId: null,
    candidateSinceMs: null,
    recoverySinceMs: null,
    lastSeenMs: null,
    candidatePosture: null,
    activePosture: null,
    criticalLogged: false,
  };
  state[label] = next;
  return next;
}

export function riskForPosture(posture?: string, lyingSec: number = 0): IncidentRisk {
  if (posture === "lying") return lyingSec >= CRITICAL_LYING_SEC ? "CRITICAL" : "HIGH";
  if (posture === "sitting") return "MEDIUM";
  return "LOW";
}

function riskPostureFor(obs: RescueIncidentObservation): IncidentPosture | null {
  if (obs.lying > 0 || obs.rescueNeeded > 0) return "lying";
  if (obs.sitting > 0) return "sitting";
  return null;
}

function patientIdFor(cam: CameraIncidentState, obs: RescueIncidentObservation): string {
  if (obs.people > 0 && obs.patientId !== "—") {
    cam.patientId = obs.patientId;
  }
  return cam.patientId ?? obs.patientId ?? "—";
}

function resetToClear(cam: CameraIncidentState) {
  cam.phase = "clear";
  cam.candidateSinceMs = null;
  cam.recoverySinceMs = null;
  cam.candidatePosture = null;
  cam.activePosture = null;
  cam.criticalLogged = false;
}

function riskDetail(posture: IncidentPosture, lyingSec: number): string {
  if (posture === "sitting") return "앉은 자세 감지";

  const sec = Math.max(0, Math.floor(lyingSec));
  if (sec >= CRITICAL_LYING_SEC) {
    return `누운 자세 10초 이상 지속 (${sec}초 경과)`;
  }
  return `누운 자세 감지 (${sec}초 경과)`;
}

function riskPostureEvent(
  obs: RescueIncidentObservation,
  patientId: string,
  posture: IncidentPosture,
): RescueIncidentEvent {
  const sec = Math.max(0, Math.floor(obs.maxLyingSec));
  return {
    kind: "위험 자세 감지 (Risk Posture)",
    cam: obs.label,
    patientId,
    risk: riskForPosture(posture, sec),
    detail: riskDetail(posture, sec),
  };
}

function escalatedEvent(obs: RescueIncidentObservation, patientId: string): RescueIncidentEvent {
  const sec = Math.max(0, Math.floor(obs.maxLyingSec));
  return {
    kind: "위험도 상승 (Risk Escalated)",
    cam: obs.label,
    patientId,
    risk: "CRITICAL",
    detail: riskDetail("lying", sec),
  };
}

function recoveredEvent(obs: RescueIncidentObservation, patientId: string): RescueIncidentEvent {
  return {
    kind: "구조 상황 해제 (Recovered)",
    cam: obs.label,
    patientId,
    risk: "LOW",
    detail: "대상 회복 또는 자세 복귀",
  };
}

export function updateRescueIncidentTimeline(
  state: RescueIncidentTimelineState,
  observations: RescueIncidentObservation[],
  nowMs: number,
): RescueIncidentEvent[] {
  const events: RescueIncidentEvent[] = [];

  for (const obs of observations) {
    const cam = getCameraState(state, obs.label);
    const visible = obs.people > 0;
    const posture = riskPostureFor(obs);
    const risk = riskForPosture(posture ?? "standing", obs.maxLyingSec);
    const shortAbsent =
      !visible && cam.lastSeenMs !== null && nowMs - cam.lastSeenMs <= ABSENT_FLICKER_GRACE_MS;

    if (visible) {
      cam.lastSeenMs = nowMs;
    }
    const patientId = patientIdFor(cam, obs);

    if (posture !== null) {
      cam.patientId = patientId;
      cam.recoverySinceMs = null;

      if (cam.phase === "active" || cam.phase === "recovering") {
        cam.phase = "active";
        cam.candidateSinceMs = null;
        cam.candidatePosture = null;

        if (cam.activePosture !== posture) {
          cam.activePosture = posture;
          cam.criticalLogged = risk === "CRITICAL";
          events.push(riskPostureEvent(obs, patientId, posture));
          continue;
        }
        if (posture === "lying" && risk === "CRITICAL" && !cam.criticalLogged) {
          cam.criticalLogged = true;
          events.push(escalatedEvent(obs, patientId));
        }
        continue;
      }

      if (cam.phase !== "risk_candidate" || cam.candidatePosture !== posture) {
        cam.phase = "risk_candidate";
        cam.candidateSinceMs = nowMs;
        cam.candidatePosture = posture;
      }
      if (cam.candidateSinceMs === null) {
        cam.candidateSinceMs = nowMs;
      }
      if (cam.phase === "risk_candidate" && nowMs - cam.candidateSinceMs >= FALL_CONFIRM_MS) {
        cam.phase = "active";
        cam.activePosture = posture;
        cam.criticalLogged = risk === "CRITICAL";
        events.push(riskPostureEvent(obs, patientId, posture));
      }
      continue;
    }

    if (cam.phase === "risk_candidate") {
      if (!shortAbsent) {
        resetToClear(cam);
      }
      continue;
    }

    if (cam.phase === "active" || cam.phase === "recovering") {
      if (!visible) {
        cam.phase = "active";
        cam.recoverySinceMs = null;
        continue;
      }

      if (cam.phase !== "recovering" || cam.recoverySinceMs === null) {
        cam.phase = "recovering";
        cam.recoverySinceMs = nowMs;
      }

      if (nowMs - cam.recoverySinceMs >= RECOVERY_CONFIRM_MS) {
        events.push(recoveredEvent(obs, patientId));
        resetToClear(cam);
      }
    }
  }

  return events;
}
