import { useCallback, useEffect, useRef, useState } from "react";
import Topbar from "../components/Topbar";
import WhepPlayer from "../components/WhepPlayer";
import KeypointOverlay from "../components/KeypointOverlay";
import { useDetectionWs } from "../hooks/useDetectionWs";
import { apiBase } from "../hooks/useApi";
import {
  type PatientEpisodeMap,
  ensurePatientEpisodeId,
  getPatientEpisodeId,
} from "./patientEpisode";
import {
  CRITICAL_LYING_SEC,
  type IncidentRisk,
  type RescueIncidentObservation,
  type RescueIncidentTimelineState,
  riskForPosture,
  updateRescueIncidentTimeline,
} from "./rescueIncidentTimeline";

/**
 * rescue-pose 관제 대시보드 (단일 화면 · tunnel DashboardPage 이식).
 *
 * - 실데이터: 등록된 전 ipcam(≤MAX_IPCAMS=16)을 NxN 그리드 라이브뷰로. 영상 = WHEP(WhepPlayer <video>),
 *   자세 keypoint = detection WebSocket(useDetectionWs) → **KeypointOverlay(스켈레톤)** 로
 *   그린다. tunnel 의 bbox 오버레이(VideoBboxOverlay)는 §3 에 따라 차용하지 않는다.
 *   카메라 미등록/백엔드 없음이면 플레이스홀더(gate-2 = 백엔드 없이 렌더).
 * - 실데이터: stat 카드(posture 집계)·구조 우선순위·상황 판단 AI·이벤트 로그(needs-rescue 전이)
 *   모두 실 WS(posture/rescueNeeded/lyingSec)에서 배선 (phase3/phase4 + 이벤트 배선).
 * - 비기능(시각): 주요 기능 버튼(119 출동/전송) — 외부통지 범위 밖 (U2, UI 전용).
 */

const C = {
  bg: "#0a0d12",
  panel: "#11151c",
  panel2: "#161b24",
  border: "#232a36",
  text: "#e6edf3",
  muted: "#8b95a5",
  red: "#e5484d",
  amber: "#e0a23b",
  green: "#3fb950",
  blue: "#4493f8",
};

// ── 목업 데이터 (백엔드 미연동, 디자인 표시용 — phase3/4 에서 실데이터로 교체) ──

type Risk = IncidentRisk;
type Posture = "LYING" | "SITTING" | "STANDING";

const RISK_COLOR: Record<Risk, string> = {
  CRITICAL: C.red,
  HIGH: "#f0883e",
  MEDIUM: C.amber,
  LOW: C.green,
};
const POSTURE_COLOR: Record<Posture, string> = {
  LYING: C.red,
  SITTING: C.amber,
  STANDING: C.green,
};

// stat 카드 — phase3 p3.c2: 사람/앉음/누움 을 실 detections 의 posture 로 집계(DashboardPage).
interface StatCard {
  icon: string;
  label: string;
  value: number;
  unit: string;
  color: string;
}

// 카메라별 집계 — CamLiveView 가 자기 detections 로 계산해 대시보드로 올린다.
// posture(소문자) + rescue(rescueNeeded/lyingSec) = WS payload(KeypointPerson, phase4 seam).
interface RescuePerson {
  cam: string;
  posture?: string;   // 소문자 'standing'|'sitting'|'lying' (표시 시 대문자 변환)
  lyingSec: number;
  rescueNeeded: boolean;
}
interface CamReport {
  people: number;
  sitting: number;
  lying: number;
  rescue: RescuePerson[];  // rescueNeeded 이거나 lying 중인 사람(우선순위 패널·경보용)
}

// 이벤트 로그 행 — 실 WS 데이터에서 파생(카메라 단위 needs-rescue 전이). 정적 mock 제거.
// per-person ID 는 프론트에서 미가용(D5) → 대상은 카메라 단위. id = 중복키 방지용 단조 증가.
interface EventRow {
  id: number;
  time: string;
  kind: string;      // "한글 (English)" 표기
  cam: string;
  patientId: string; // 카메라별 active episode ID. 감지 gap 은 10초까지 같은 ID 유지.
  risk: Risk;
  detail: string;
}
const MAX_EVENTS = 20;
const PAGE_SIZE = 5;
function fmtClock(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// ── 실데이터: ipcam 라이브 프레임 ────────────────────────────────────

interface IpCam {
  id: number;
  name: string;
  stream_key: string;
  sync_pose?: boolean;
}

// CAM 슬롯 label — 위치 기반 `CAM 01`… (등록 순서). reportBySlot 집계 키이자 React key.
function camLabel(i: number): string {
  return `CAM ${String(i + 1).padStart(2, "0")}`;
}

/**
 * 등록된 ipcam 한 대의 라이브뷰 — 영상=WHEP(<video>), 자세=detection WebSocket.
 * WhepPlayer 와 KeypointOverlay 가 같은 videoRef 를 공유해 canvas 를 video 위에 겹친다
 * (rtsp-keypoint CameraGrid 와 동일 패턴). det 소비 = KeypointPerson{keypoints,model,posture?}.
 */
function CamLiveView({
  streamKey,
  label,
  syncPose,
  onReport,
  onRescueEnd,
}: {
  streamKey: string;
  label: string;
  syncPose: boolean;
  onReport: (label: string, r: CamReport) => void;
  onRescueEnd: (label: string, reasons: string[]) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [detectionActive, setDetectionActive] = useState(false);
  // 대시보드는 자세를 항상 시각화 — active=true 로 WS 연결(백엔드는 추론 ON 일 때만 emit).
  // backend reload 후 메모리 기반 per-source 설정이 비면 global 모델로 초기화한 뒤 WS 를 연다.
  useEffect(() => {
    let cancelled = false;
    async function ensureInference() {
      setDetectionActive(false);
      try {
        const stateRes = await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`);
        const state = stateRes.ok ? await stateRes.json() : {};
        const currentModels = Array.isArray(state.models) ? state.models : [];
        if (state.enabled === true && currentModels.length > 0) {
          if (!cancelled) setDetectionActive(true);
          return;
        }

        const globalRes = await fetch(`${apiBase()}/api/inference/config`);
        const global = globalRes.ok ? await globalRes.json() : {};
        const model = typeof global.model === "string" && global.model
          ? global.model
          : "yolo26n-pose.pt";
        const body: Record<string, unknown> = {
          enabled: true,
          models: currentModels.length > 0 ? currentModels : [model],
        };
        if (typeof global.conf_threshold === "number") {
          body.conf_threshold = global.conf_threshold;
        }
        const putRes = await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const updated = putRes.ok ? await putRes.json() : body;
        const models = Array.isArray(updated.models) ? updated.models : body.models;
        if (!cancelled) {
          setDetectionActive(updated.enabled === true && Array.isArray(models) && models.length > 0);
        }
      } catch {
        if (!cancelled) setDetectionActive(true);
      }
    }
    ensureInference();
    return () => {
      cancelled = true;
    };
  }, [streamKey]);
  const handleEvents = useCallback((reasons: string[]) => onRescueEnd(label, reasons), [label, onRescueEnd]);
  const { items, frameW, frameH, captureTs, detFps, annotatedFrame } = useDetectionWs(
    streamKey,
    detectionActive,
    handleEvents,
  );
  const syncedPose = detectionActive && syncPose;
  const [live, setLive] = useState(false);
  const onFps = useCallback((f: number) => setLive(f > 0), []);

  useEffect(() => {
    if (syncedPose) setLive(detFps > 0);
  }, [syncedPose, detFps]);

  // 실 집계 — 이 카메라 detections 의 posture 카운트 + rescue 대상 리스트를 부모(대시보드)로 올린다.
  // people = 감지된 사람 수(posture 무관), sitting/lying = 해당 posture 인원, rescue = 판정 대상.
  let sitting = 0;
  let lying = 0;
  const rescue: RescuePerson[] = [];
  for (const p of items) {
    if (p.posture === "sitting") sitting++;
    else if (p.posture === "lying") lying++;
    const lyingNow = p.posture === "lying";
    if (p.rescueNeeded || lyingNow || (p.lyingSec ?? 0) > 0) {
      rescue.push({
        cam: label,
        posture: p.posture,
        lyingSec: p.lyingSec ?? 0,
        rescueNeeded: p.rescueNeeded === true || lyingNow,
      });
    }
  }
  const people = items.length;
  // items 갱신마다 리포트 — 부모가 초 단위 signature 로 dedup 해 과도한 re-render 를 막는다.
  useEffect(() => {
    onReport(label, { people, sitting, lying, rescue });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- people/sitting/lying/rescue 는 items 파생값 → items ref 로 gate.
  }, [items, label, onReport]);
  // 언마운트(카메라 제거/플레이스홀더 전환) 시 이 슬롯 집계를 0/빈 리스트로 정리.
  useEffect(
    () => () => onReport(label, { people: 0, sitting: 0, lying: 0, rescue: [] }),
    [label, onReport],
  );

  return (
    <div style={s.camCard}>
      <div style={s.camHeader}>
        <span style={s.camLabel}>{label}</span>
        <span style={{ ...s.liveTag, color: live ? C.green : C.muted }}>
          ● {live ? "LIVE" : "신호 없음"}
        </span>
      </div>
      <div style={s.camBody}>
        {syncedPose ? (
          annotatedFrame ? (
            <img src={annotatedFrame} alt="" style={s.syncedFrame} />
          ) : (
            <span style={s.camPlaceholder}>동기화 대기</span>
          )
        ) : (
          <>
            <WhepPlayer streamKey={streamKey} videoRef={videoRef} onFps={onFps} />
            <KeypointOverlay
              videoRef={videoRef}
              detections={items}
              captureTs={captureTs}
              frameW={frameW}
              frameH={frameH}
            />
          </>
        )}
      </div>
    </div>
  );
}

/** 카메라 미등록/백엔드 없음 슬롯 플레이스홀더. */
function CamPlaceholder({ label }: { label: string }) {
  return (
    <div style={s.camCard}>
      <div style={s.camHeader}>
        <span style={s.camLabel}>{label}</span>
        <span style={{ ...s.liveTag, color: C.muted }}>● 미등록</span>
      </div>
      <div style={{ ...s.camBody, ...s.camPlaceholder }}>
        등록된 카메라 없음 — 설정에서 RTSP 추가
      </div>
    </div>
  );
}

// ── 메인 대시보드 ────────────────────────────────────────────────────

function DashboardPage() {
  const [cams, setCams] = useState<IpCam[]>([]);
  const [now, setNow] = useState(new Date());
  // 카메라별 집계 — CamLiveView 가 onReport 로 올린 값을 슬롯 label 별로 저장.
  const [reportBySlot, setReportBySlot] = useState<Record<string, CamReport>>({});
  const sigRef = useRef<Record<string, string>>({});
  const reportCam = useCallback((label: string, r: CamReport) => {
    // 초 단위 signature 로 dedup — lyingSec 연속 변화로 인한 과도한 re-render 방지(초 넘을 때만 갱신).
    const sig =
      `${r.people}|${r.sitting}|${r.lying}|` +
      r.rescue
        .map((x) => `${x.rescueNeeded ? 1 : 0}:${Math.floor(x.lyingSec)}:${x.posture ?? ""}`)
        .join(",");
    if (sigRef.current[label] === sig) return;
    sigRef.current[label] = sig;
    setReportBySlot((prev) => ({ ...prev, [label]: r }));
  }, []);

  // rescue 종료 이벤트는 프론트 즉시 전이 감지에서 처리한다. backend 이벤트는 중복 방지용으로 소비만 한다.
  const handleRescueEnd = useCallback((label: string, reasons: string[]) => {
    void label;
    void reasons;
  }, []);

  // 이벤트 로그(실데이터) — raw 자세 변화가 아니라 카메라별 구조 사건 타임라인만 기록한다.
  const [events, setEvents] = useState<EventRow[]>([]);
  const [page, setPage] = useState(0);
  const eventIdRef = useRef(0);
  const patientSeqRef = useRef(0);
  const patientEpisodesRef = useRef<PatientEpisodeMap>({});
  const incidentTimelineRef = useRef<RescueIncidentTimelineState>({});
  const nextPatientId = () => `P-${String(++patientSeqRef.current).padStart(2, "0")}`;
  const patientIdForSlot = (label: string, people: number, nowMs: number) => {
    if (people > 0) {
      return ensurePatientEpisodeId(patientEpisodesRef.current, label, nowMs, nextPatientId);
    }
    return getPatientEpisodeId(patientEpisodesRef.current, label, nowMs) ?? "—";
  };

  // 등록된 ipcam fetch (실데이터). 백엔드 없으면 빈 배열(gate-2 = graceful).
  useEffect(() => {
    fetch(`${apiBase()}/api/ipcams`)
      .then((r) => r.json())
      .then((data: IpCam[]) => setCams(data))
      .catch(() => setCams([]));
  }, []);

  // 실시간 시계
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // 구조 사건 타임라인 — UI 표시는 즉시 갱신하되, 로그는 안정된 사건 시작/종료만 append 한다.
  useEffect(() => {
    const nowMs = now.getTime();
    const observations: RescueIncidentObservation[] = Object.entries(reportBySlot).map(([label, r]) => {
      const rescueNeeded = r.rescue.filter((x) => x.rescueNeeded).length;
      const maxLyingSec = Math.floor(
        Math.max(0, ...r.rescue.filter((x) => x.rescueNeeded).map((x) => x.lyingSec)),
      );
      return {
        label,
        people: r.people,
        sitting: r.sitting,
        lying: r.lying,
        rescueNeeded,
        maxLyingSec,
        patientId: patientIdForSlot(label, r.people, nowMs),
      };
    });
    const timelineEvents = updateRescueIncidentTimeline(incidentTimelineRef.current, observations, nowMs);
    if (!timelineEvents.length) return;

    const time = fmtClock(now);
    const rows: EventRow[] = timelineEvents.map((e) => ({
      id: ++eventIdRef.current,
      time,
      kind: e.kind,
      cam: e.cam,
      patientId: e.patientId,
      risk: e.risk,
      detail: e.detail,
    }));
    setEvents((prev) => [...rows, ...prev].slice(0, MAX_EVENTS));
  }, [reportBySlot, now]);

  const DOW = ["일", "월", "화", "수", "목", "금", "토"];
  const dateStr =
    `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, "0")}.${String(now.getDate()).padStart(2, "0")} (${DOW[now.getDay()]})`;
  const timeStr =
    `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;

  // 카메라 뷰 = 항상 2분할(2열). 등록 0/1/2개 무관 최소 2슬롯 유지 — 빈 슬롯은 CamPlaceholder.
  // cams 가 3개+ 여도 2열 고정(여러 행). 실 카메라만 CamLiveView(WS/rescue), 나머지는 자리표시.
  const camSlots = Math.max(2, cams.length);

  // 이벤트 로그 페이지네이션 — 페이지당 PAGE_SIZE, page 는 저장범위 초과 시 clamp.
  const evtPages = Math.max(1, Math.ceil(events.length / PAGE_SIZE));
  const evtPage = Math.min(page, evtPages - 1);
  const evtSlice = events.slice(evtPage * PAGE_SIZE, evtPage * PAGE_SIZE + PAGE_SIZE);

  // 전 카메라 합산 → stat 카드 + 구조 우선순위 + 상황판단 + 경보의 실데이터(mock 대체).
  const totals = { people: 0, sitting: 0, lying: 0 };
  const rescuePersons: RescuePerson[] = [];
  const priorityPersons: RescuePerson[] = [];
  for (const [label, r] of Object.entries(reportBySlot)) {
    totals.people += r.people;
    totals.sitting += r.sitting;
    totals.lying += r.lying;
    rescuePersons.push(...r.rescue);
    priorityPersons.push(...r.rescue);
    const hasSittingPriority = r.rescue.some((p) => p.posture === "sitting");
    if (r.sitting > 0 && !hasSittingPriority) {
      priorityPersons.push({
        cam: label,
        posture: "sitting",
        lyingSec: 0,
        rescueNeeded: false,
      });
    }
  }
  const needsRescue = rescuePersons.filter((r) => r.rescueNeeded);
  const needsRescueCount = needsRescue.length;
  const riskWeight: Record<Risk, number> = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
  // 우선순위: 위험도 높은 순, 같은 위험도면 누운 시간 긴 순.
  const priority = [...priorityPersons].sort(
    (a, b) =>
      riskWeight[riskForPosture(b.posture, b.lyingSec)] -
        riskWeight[riskForPosture(a.posture, a.lyingSec)] ||
      b.lyingSec - a.lyingSec,
  );
  const top = priority[0];

  const stats: StatCard[] = [
    { icon: "👥", label: "총 인원", value: totals.people, unit: "명", color: C.text },
    { icon: "🧎", label: "앉은 자세", value: totals.sitting, unit: "명", color: C.amber },
    { icon: "🛌", label: "누운 자세", value: totals.lying, unit: "명", color: C.red },
    { icon: "🚨", label: "구조 필요", value: needsRescueCount, unit: "명", color: C.red },
  ];

  return (
    <div style={s.root}>
      {/* ── 상단 바 (공용) ── */}
      <Topbar
        active="monitor"
        right={
          <>
            <button style={s.btnAlertTop}>🚨 긴급 구조 요청</button>
            <div style={s.clock}>
              <div style={s.clockDate}>{dateStr}</div>
              <div style={s.clockTime}>{timeStr}</div>
            </div>
          </>
        }
      />

      {/* ── 본문 그리드 ── */}
      <div style={s.body}>
        {/* 좌측 (메인) */}
        <div style={s.colMain}>
          {/* stat 카드 4개 — 사람/앉음/누움 + 구조필요 실집계 */}
          <div style={s.statRow}>
            {stats.map((st) => (
              <div key={st.label} style={s.statCard}>
                <span style={s.statIcon}>{st.icon}</span>
                <div>
                  <div style={s.statLabel}>{st.label}</div>
                  <div style={{ ...s.statValue, color: st.color }}>
                    {st.value} <span style={s.statUnit}>{st.unit}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 전-카메라 라이브뷰 그리드 (실데이터 · 스켈레톤 오버레이 · NxN) */}
          <div style={{ ...s.camRow, gridTemplateColumns: "repeat(2, 1fr)" }}>
            {Array.from({ length: camSlots }).map((_, i) => {
              const cam = cams[i];
              const label = camLabel(i);
              return cam ? (
                <CamLiveView
                  key={label}
                  streamKey={cam.stream_key}
                  label={label}
                  syncPose={cam.sync_pose === true}
                  onReport={reportCam}
                  onRescueEnd={handleRescueEnd}
                />
              ) : (
                <CamPlaceholder key={`ph-${i}`} label={label} />
              );
            })}
          </div>

          {/* 이벤트 로그 (실데이터 — needs-rescue 전이, 최근 순) */}
          <div style={s.panel}>
            <div style={s.panelHeader}>
              <span style={s.panelTitle}>이벤트 로그</span>
              <div style={s.eventHeaderActions}>
                {events.length > PAGE_SIZE && (
                  <div style={s.pager}>
                    <span style={s.pagerBr}>&lt;</span>
                    {Array.from({ length: evtPages }).map((_, i) => (
                      <span key={i} style={s.pagerItem}>
                        {i > 0 && <span style={s.pagerSep}>, </span>}
                        <span onClick={() => setPage(i)} style={{ ...s.pagerNum, ...(i === evtPage ? s.pagerNumOn : {}) }}>{i + 1}</span>
                      </span>
                    ))}
                    <span style={s.pagerBr}>&gt;</span>
                  </div>
                )}
                <span style={s.viewAll} onClick={() => { setEvents([]); setPage(0); }}>이벤트 삭제</span>
              </div>
            </div>
            <table style={s.table}>
              <colgroup>
                <col style={{ width: "86px" }} />
                <col style={{ width: "220px" }} />
                <col style={{ width: "84px" }} />
                <col style={{ width: "86px" }} />
                <col style={{ width: "100px" }} />
                <col />
                <col style={{ width: "64px" }} />
              </colgroup>
              <thead>
                <tr>
                  {["시간", "이벤트", "카메라", "환자 ID", "위험도", "상세 내용", ""].map((h, i) => (
                    <th key={i} style={s.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {evtSlice.map((e) => (
                  <tr key={e.id} style={s.trow}>
                    <td style={s.tdTime}>{e.time}</td>
                    <td style={s.td}>
                      <span style={s.eventCell}><EventIcon risk={e.risk} /><span style={s.cellText}>{e.kind}</span></span>
                    </td>
                    <td style={{ ...s.td, whiteSpace: "nowrap" }}>{e.cam}</td>
                    <td style={{ ...s.td, color: C.muted, whiteSpace: "nowrap" }}>{e.patientId}</td>
                    <td style={s.td}><RiskBadge risk={e.risk} /></td>
                    <td style={{ ...s.td, color: C.muted }} title={e.detail}>{e.detail}</td>
                    <td style={s.td}><div style={s.thumb} title="스냅샷 (배선 예정)"><div style={s.thumbIcon} /></div></td>
                  </tr>
                ))}
                {Array.from({ length: PAGE_SIZE - evtSlice.length }).map((_, i) => (
                  <tr key={`empty-${i}`} style={s.trow}>
                    <td colSpan={7} style={s.td} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* 우측 (사이드) */}
        <div style={s.colSide}>
          {/* 구조 우선순위 (실데이터 — rescueNeeded 먼저, 누운 시간 긴 순) */}
          <div style={{ ...s.panel, ...s.prioPanel }}>
            <div style={s.panelHeader}>
              <span style={s.panelTitle}>구조 우선순위 <span style={s.panelTitleSub}>(AI 판단)</span></span>
            </div>
            <div style={s.prioList}>
              {priority.length === 0 ? (
                <div style={s.prioEmpty} aria-label="구조 필요 대상 없음" />
              ) : (
                priority.map((p, i) => {
                  const secs = Math.floor(p.lyingSec);
                  const posture = p.posture === "sitting" ? "sitting" : p.posture === "standing" ? "standing" : "lying";
                  const risk: Risk = riskForPosture(posture, secs);
                  const postureUp = posture.toUpperCase() as Posture;
                  const detail =
                    posture === "sitting"
                      ? "앉은 자세 감지 · 관찰 필요"
                      : secs >= CRITICAL_LYING_SEC
                        ? `누운 자세 10초 이상 지속 (${secs}초 경과) · 즉시 구조 필요`
                        : `누운 자세 감지 (${secs}초 경과) · 위험도 HIGH`;
                  return (
                    <div key={`${p.cam}-${i}`} style={s.prioItem}>
                      <span style={{ ...s.prioRank, background: RISK_COLOR[risk] }}>{i + 1}</span>
                      <div style={{ flex: 1 }}>
                        <div style={s.prioTop}>
                          <span style={s.prioPid}>{p.cam}</span>
                          <span style={{ ...s.postureTag, color: POSTURE_COLOR[postureUp], borderColor: POSTURE_COLOR[postureUp] }}>
                            {postureUp}
                          </span>
                          <span style={s.prioScore}>
                            {posture === "sitting" ? "앉은 자세" : `누운 지 ${secs}s`}
                          </span>
                          <RiskBadge risk={risk} />
                        </div>
                        <div style={s.prioDetail}>{detail}</div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* 상황 판단 결과 (실데이터) */}
          <div style={s.panel}>
            <div style={s.panelTitle}>상황 판단 결과 <span style={s.panelTitleSub}>(AI)</span></div>
            <p style={s.aiText}>현재 총 {totals.people}명의 인원이 탐지되었습니다.</p>
            <ul style={s.aiList}>
              <li>누워있는 대상: {totals.lying}명</li>
              <li>구조 필요: {needsRescueCount}명</li>
              <li>앉은 자세: {totals.sitting}명</li>
            </ul>
            <p style={s.aiHighlight}>
              {needsRescueCount > 0 && top
                ? `최우선 구조 대상은 ${top.cam} (누운 지 ${Math.floor(top.lyingSec)}초) 입니다.`
                : top?.posture === "sitting"
                  ? `최우선 관찰 대상은 ${top.cam} (앉은 자세) 입니다.`
                  : "현재 구조가 필요한 대상이 없습니다."}
            </p>
          </div>

          {/* 주요 기능 (mock) */}
          <div style={s.panel}>
            <div style={s.panelTitle}>주요 기능</div>
            <div style={s.funcGrid}>
              <FuncBtn color={C.red} title="119 출동 요청" sub="자동 구조대 지원 요청" icon="🚑" />
              <FuncBtn color={C.amber} title="구조 우선순위 전송" sub="대상 정보 · 위치 전송" icon="🚒" />
              <FuncBtn color={C.blue} title="현장 스냅샷 저장" sub="현재 화면 저장" icon="📷" />
              <FuncBtn color={C.green} title="상황 보고서 생성" sub="AI 분석 리포트 생성" icon="📄" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function RiskBadge({ risk }: { risk: Risk }) {
  return (
    <span style={{ ...s.riskBadge, color: RISK_COLOR[risk], borderColor: RISK_COLOR[risk] }}>
      {risk}
    </span>
  );
}

// 이벤트 심각도 아이콘 — CRITICAL 빨강 / HIGH·MEDIUM 호박 채운 삼각형(!), LOW 파랑 채운 원(i).
function EventIcon({ risk }: { risk: Risk }) {
  if (risk === "LOW") {
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" style={{ flexShrink: 0 }} aria-hidden>
        <circle cx="9" cy="9" r="8" fill={C.blue} />
        <text x="9" y="13.2" textAnchor="middle" fontSize="11" fontWeight="700" fontStyle="italic" fill="#fff">i</text>
      </svg>
    );
  }
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" style={{ flexShrink: 0 }} aria-hidden>
      <path d="M9 1.5 L17 16 L1 16 Z" fill={RISK_COLOR[risk]} />
      <text x="9" y="14.6" textAnchor="middle" fontSize="10.5" fontWeight="800" fill="#0a0d12">!</text>
    </svg>
  );
}
function FuncBtn({ color, title, sub, icon }: { color: string; title: string; sub: string; icon: string }) {
  return (
    <button style={{ ...s.funcBtn, borderColor: color }}>
      <span style={{ ...s.funcIcon, color }}>{icon}</span>
      <span style={s.funcTitle}>{title}</span>
      <span style={s.funcSub}>{sub}</span>
    </button>
  );
}

const s: Record<string, React.CSSProperties> = {
  root: { minHeight: "100vh", overflowX: "hidden", background: C.bg, color: C.text },

  // 상단 바 우측 슬롯
  btnAlertTop: { padding: "0.48rem 0.95rem", borderRadius: "8px", border: "none", background: C.red, color: "#fff", fontWeight: 700, fontSize: "0.86rem", cursor: "pointer" },
  clock: { textAlign: "right", fontFamily: "monospace" },
  clockDate: { fontSize: "0.72rem", color: C.muted },
  clockTime: { fontSize: "1.05rem", fontWeight: 700 },

  // 본문
  body: { minHeight: 0, display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(320px, 360px)", gap: "var(--rp-gap)", padding: "var(--rp-page-pad)", alignItems: "stretch" },
  colMain: { display: "flex", flexDirection: "column", gap: "var(--rp-gap)", minWidth: 0, minHeight: 0 },
  colSide: { display: "flex", flexDirection: "column", gap: "var(--rp-gap)", minHeight: 0, alignSelf: "stretch" },

  // stat 카드
  statRow: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "var(--rp-gap-sm)", flexShrink: 0 },
  statCard: { display: "flex", alignItems: "center", gap: "0.68rem", minHeight: "clamp(58px, 6.8vh, 76px)", padding: "0.72rem 0.85rem", background: C.panel, border: `1px solid ${C.border}`, borderRadius: "10px" },
  statIcon: { fontSize: "1.38rem" },
  statLabel: { fontSize: "0.8rem", color: C.muted },
  statValue: { fontSize: "1.45rem", fontWeight: 800, lineHeight: 1.05 },
  statUnit: { fontSize: "0.85rem", fontWeight: 500, color: C.muted },

  // CAM
  camRow: { display: "grid", gap: "var(--rp-gap-sm)", minHeight: 0, flexShrink: 1 },
  camCard: { minHeight: 0, background: C.panel, border: `1px solid ${C.border}`, borderRadius: "10px", overflow: "hidden" },
  camHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", minHeight: "30px", padding: "0.35rem 0.65rem" },
  camLabel: { fontSize: "0.85rem", fontWeight: 600 },
  liveTag: { fontSize: "0.75rem", fontWeight: 700 },
  camBody: { position: "relative", aspectRatio: "16 / 9", maxHeight: "var(--rp-cam-body-max-h)", background: "#000", display: "flex", alignItems: "center", justifyContent: "center" },
  syncedFrame: { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" as const },
  camPlaceholder: { color: C.muted, fontSize: "0.85rem" },

  // 공용 패널
  panel: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: "10px", padding: "var(--rp-panel-pad)" },
  panelHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.55rem" },
  panelTitle: { fontSize: "0.95rem", fontWeight: 700 },
  panelTitleSub: { fontSize: "0.78rem", fontWeight: 500, color: C.muted },
  linkBtn: { fontSize: "0.78rem", color: C.muted, cursor: "pointer" },

  // 테이블
  table: { width: "100%", borderCollapse: "collapse", tableLayout: "fixed" },
  th: { height: "var(--rp-event-head-h)", textAlign: "left", padding: "0 0.6rem", fontSize: "0.72rem", color: C.muted, borderBottom: `1px solid ${C.border}`, fontWeight: 500, letterSpacing: "0.02em", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
  trow: { height: "var(--rp-event-row-h)", borderBottom: `1px solid ${C.border}` },
  td: { height: "var(--rp-event-row-h)", maxHeight: "var(--rp-event-row-h)", padding: "0 0.6rem", fontSize: "0.8rem", verticalAlign: "middle", color: C.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
  tdTime: { height: "var(--rp-event-row-h)", maxHeight: "var(--rp-event-row-h)", padding: "0 0.6rem", fontSize: "0.8rem", verticalAlign: "middle", color: C.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontVariantNumeric: "tabular-nums" },

  eventHeaderActions: { display: "inline-flex", alignItems: "center", gap: "0.75rem", minWidth: 0 },
  viewAll: { fontSize: "0.8rem", fontWeight: 600, color: C.blue, cursor: "pointer", whiteSpace: "nowrap" },
  pager: { display: "inline-flex", alignItems: "center", justifyContent: "center", gap: "0.1rem", fontSize: "0.78rem", color: C.muted, fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" },
  pagerBr: { color: C.muted, opacity: 0.6, margin: "0 0.15rem" },
  pagerItem: { display: "inline-flex", alignItems: "center" },
  pagerSep: { color: C.muted, opacity: 0.5 },
  pagerNum: { cursor: "pointer", padding: "0.1rem 0.3rem", borderRadius: "5px", color: C.muted },
  pagerNumOn: { color: C.text, fontWeight: 700 },
  eventCell: { display: "inline-flex", alignItems: "center", gap: "0.5rem", maxWidth: "100%", color: C.text, fontWeight: 500, overflow: "hidden", whiteSpace: "nowrap" },
  cellText: { minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  thumb: { width: "44px", height: "26px", borderRadius: "6px", background: `linear-gradient(135deg, ${C.panel2}, ${C.bg})`, border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center" },
  thumbIcon: { width: "18px", height: "14px", borderRadius: "2px", border: `1.5px solid ${C.muted}`, opacity: 0.5 },
  // 위험도 배지
  riskBadge: { display: "inline-block", padding: "0.22rem 0.6rem", borderRadius: "6px", border: "1px solid", fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.02em" },

  // 우선순위
  prioPanel: { flex: 1, minHeight: 0, display: "flex", flexDirection: "column" },
  prioList: { flex: 1, minHeight: 0, display: "flex", flexDirection: "column", gap: "0.5rem", overflowY: "auto" },
  prioEmpty: { flex: 1, minHeight: "clamp(140px, 18vh, 220px)", background: C.panel2, borderRadius: "8px", border: `1px solid ${C.border}`, opacity: 0.55 },
  prioItem: { display: "flex", gap: "0.5rem", padding: "0.5rem", background: C.panel2, borderRadius: "8px", border: `1px solid ${C.border}` },
  prioRank: { width: "22px", height: "22px", borderRadius: "6px", color: "#fff", fontWeight: 800, fontSize: "0.8rem", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 },
  prioTop: { display: "flex", alignItems: "center", gap: "0.4rem", flexWrap: "wrap" },
  prioPid: { fontSize: "0.9rem", fontWeight: 700 },
  postureTag: { fontSize: "0.62rem", fontWeight: 700, border: "1px solid", borderRadius: "4px", padding: "0.05rem 0.3rem" },
  prioScore: { fontSize: "0.78rem", color: C.muted },
  prioDetail: { fontSize: "0.72rem", color: C.muted, marginTop: "0.25rem" },

  // 상황판단 AI
  aiText: { fontSize: "0.85rem", margin: "0 0 0.5rem" },
  aiList: { margin: 0, paddingLeft: "1.1rem", fontSize: "0.8rem", color: C.muted, lineHeight: 1.55 },
  aiHighlight: { fontSize: "0.85rem", fontWeight: 700, color: C.red, marginTop: "0.5rem" },

  // 주요 기능
  funcGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" },
  funcBtn: { display: "flex", flexDirection: "column", gap: "0.16rem", padding: "0.62rem", background: C.panel2, border: "1px solid", borderRadius: "8px", color: C.text, cursor: "pointer", textAlign: "left" },
  funcIcon: { fontSize: "1.05rem" },
  funcTitle: { fontSize: "0.85rem", fontWeight: 700 },
  funcSub: { fontSize: "0.68rem", color: C.muted },
};

export default DashboardPage;
