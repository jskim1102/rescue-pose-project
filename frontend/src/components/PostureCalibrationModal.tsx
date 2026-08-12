import { useEffect, useRef, useState } from "react";

import { apiBase } from "../hooks/useApi";
import { useDetectionWs } from "../hooks/useDetectionWs";
import {
  anchorDistancesFromWorldPoints,
  convexHullPoints,
  floorWorldPointsFromAnchorDistances,
  mapClientPointToVideo,
  selectStandingReferenceAtPoint,
  type CalibrationPoint,
  type FloorAnchorDistances,
  type PostureCalibrationPayload,
  type StandingReferencePayload,
} from "../utils/postureCalibration";
import KeypointOverlay from "./KeypointOverlay";
import Modal from "./Modal";
import WhepPlayer from "./WhepPlayer";


interface Props {
  open: boolean;
  onClose: () => void;
  cameraId: number;
  cameraName: string;
  streamKey: string;
  inferenceActive: boolean;
  hasModels: boolean;
  onEnableInference: () => void;
}

interface CalibrationState {
  enabled: boolean;
  calibration: PostureCalibrationPayload | null;
}

interface FrameSize {
  width: number;
  height: number;
}

type Step = "floor" | "standing";
type AnchorDistanceInputs = Record<keyof FloorAnchorDistances, string>;

const FLOOR_POINT_LABELS = [
  "A · 바닥의 기준점",
  "B · A에서 떨어진 바닥점",
  "C · A-B 선 밖의 바닥점",
  "D · A-B 선 밖의 또 다른 바닥점",
];
const EMPTY_ANCHOR_DISTANCES: AnchorDistanceInputs = { ab: "", ac: "", bc: "", ad: "", bd: "" };
const DISTANCE_FIELDS: Array<{ key: keyof FloorAnchorDistances; label: string; requiredPoints: number }> = [
  { key: "ab", label: "A-B", requiredPoints: 2 },
  { key: "ac", label: "A-C", requiredPoints: 3 },
  { key: "bc", label: "B-C", requiredPoints: 3 },
  { key: "ad", label: "A-D", requiredPoints: 4 },
  { key: "bd", label: "B-D", requiredPoints: 4 },
];
const MEASUREMENT_EDGES: Array<{ first: number; second: number; label: string }> = [
  { first: 0, second: 1, label: "AB" },
  { first: 0, second: 2, label: "AC" },
  { first: 1, second: 2, label: "BC" },
  { first: 0, second: 3, label: "AD" },
  { first: 1, second: 3, label: "BD" },
];

async function responseDetail(response: Response, fallback: string): Promise<string> {
  const body = await response.json().catch(() => null) as { detail?: unknown } | null;
  return typeof body?.detail === "string" ? body.detail : fallback;
}

function distanceInputsFromWorldPoints(points: CalibrationPoint[]): AnchorDistanceInputs {
  const distances = anchorDistancesFromWorldPoints(points);
  return Object.fromEntries(
    Object.entries(distances).map(([key, value]) => [key, value.toFixed(2)]),
  ) as AnchorDistanceInputs;
}

function numericAnchorDistances(inputs: AnchorDistanceInputs): FloorAnchorDistances {
  return {
    ab: Number(inputs.ab),
    ac: Number(inputs.ac),
    bc: Number(inputs.bc),
    ad: Number(inputs.ad),
    bd: Number(inputs.bd),
  };
}

function pointInPolygon(point: CalibrationPoint, polygon: CalibrationPoint[]): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
    const currentPoint = polygon[index];
    const previousPoint = polygon[previous];
    if (!currentPoint || !previousPoint) continue;
    const intersects = (
      (currentPoint[1] > point[1]) !== (previousPoint[1] > point[1])
      && point[0] < (
        (previousPoint[0] - currentPoint[0])
        * (point[1] - currentPoint[1])
        / (previousPoint[1] - currentPoint[1])
        + currentPoint[0]
      )
    );
    if (intersects) inside = !inside;
  }
  return inside;
}

function PostureCalibrationModal({
  open,
  onClose,
  cameraId,
  cameraName,
  streamKey,
  inferenceActive,
  hasModels,
  onEnableInference,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [step, setStep] = useState<Step>("floor");
  const [frameSize, setFrameSize] = useState<FrameSize | null>(null);
  const [videoSize, setVideoSize] = useState<FrameSize | null>(null);
  const [floorPoints, setFloorPoints] = useState<CalibrationPoint[]>([]);
  const [anchorDistanceInputs, setAnchorDistanceInputs] = useState<AnchorDistanceInputs>(EMPTY_ANCHOR_DISTANCES);
  const [personHeightM, setPersonHeightM] = useState("1.70");
  const [standingReferences, setStandingReferences] = useState<StandingReferencePayload[]>([]);
  const [enabled, setEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const { items, frameW, frameH } = useDetectionWs(streamKey, open && inferenceActive);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !open) return;
    const update = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        setVideoSize({ width: video.videoWidth, height: video.videoHeight });
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    update();
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setMessage(null);
    fetch(`${apiBase()}/api/ipcams/${cameraId}/calibration`)
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseDetail(response, "보정값을 불러오지 못했습니다"));
        return response.json() as Promise<CalibrationState>;
      })
      .then((state) => {
        if (cancelled) return;
        setEnabled(state.enabled);
        if (!state.calibration) return;
        const calibration = state.calibration;
        setFrameSize({ width: calibration.frame_width, height: calibration.frame_height });
        setFloorPoints(calibration.floor_image_points.slice(0, 4));
        setStandingReferences(calibration.standing_references);
        setAnchorDistanceInputs(distanceInputsFromWorldPoints(calibration.floor_world_points));
        setPersonHeightM(calibration.standing_references[0]?.height_m.toFixed(2) ?? "1.70");
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setMessage({ kind: "error", text: error instanceof Error ? error.message : "보정값을 불러오지 못했습니다" });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, cameraId]);

  // 새 보정은 현재 WHEP 원본 해상도를 좌표계로 쓴다. 기존 보정은 저장된 좌표계를 유지한다.
  useEffect(() => {
    if (!loading && !frameSize && videoSize) setFrameSize(videoSize);
  }, [frameSize, loading, videoSize]);

  const allDistancesEntered = Object.values(anchorDistanceInputs).every((value) => value.trim() !== "");
  let computedFloorWorldPoints: CalibrationPoint[] | null = null;
  let floorGeometryError = "";
  try {
    computedFloorWorldPoints = floorWorldPointsFromAnchorDistances(
      floorPoints,
      numericAnchorDistances(anchorDistanceInputs),
    );
  } catch (error) {
    computedFloorWorldPoints = null;
    if (floorPoints.length === 4 && allDistancesEntered) {
      floorGeometryError = error instanceof Error ? error.message : "기준점 거리 조합을 확인하세요";
    }
  }
  const floorHull = convexHullPoints(floorPoints);

  const setError = (text: string) => setMessage({ kind: "error", text });

  const clickedCalibrationPoint = (event: React.MouseEvent<SVGSVGElement>): CalibrationPoint | null => {
    const video = videoRef.current;
    if (!video || !videoSize || !frameSize) return null;
    const naturalPoint = mapClientPointToVideo(
      { x: event.clientX, y: event.clientY },
      video.getBoundingClientRect(),
      videoSize.width,
      videoSize.height,
    );
    if (!naturalPoint) return null;
    return [
      naturalPoint[0] * frameSize.width / videoSize.width,
      naturalPoint[1] * frameSize.height / videoSize.height,
    ];
  };

  const handleVideoClick = (event: React.MouseEvent<SVGSVGElement>) => {
    const point = clickedCalibrationPoint(event);
    if (!point || !frameSize) {
      setError("검은 여백이 아니라 실제 영상 안을 클릭하세요");
      return;
    }
    if (step === "floor") {
      if (floorPoints.length >= 4) {
        setError("바닥점 4개를 이미 찍었습니다. 다시 찍기를 누르면 수정할 수 있습니다");
        return;
      }
      setFloorPoints((current) => [...current, point]);
      setMessage(null);
      return;
    }

    if (!inferenceActive) {
      setError("사람을 자동 선택하려면 먼저 추론을 켜세요");
      return;
    }
    const heightM = Number(personHeightM);
    if (!Number.isFinite(heightM) || heightM < 0.5 || heightM > 2.5) {
      setError("기준 사람의 실제 키를 0.5~2.5m로 입력하세요");
      return;
    }
    const reference = selectStandingReferenceAtPoint(
      items,
      point,
      { frameWidth: frameW, frameHeight: frameH },
      { calibrationWidth: frameSize.width, calibrationHeight: frameSize.height },
      heightM,
    );
    if (!reference) {
      setError("사람의 몸을 클릭하세요. 발목을 포함한 키포인트가 보여야 합니다");
      return;
    }
    if (!pointInPolygon(reference.foot_px, floorHull)) {
      setError("선택한 사람의 발이 지정한 바닥 영역 밖에 있습니다");
      return;
    }
    if (standingReferences.length >= 20) {
      setError("기립 기준은 최대 20개까지 저장할 수 있습니다");
      return;
    }
    setStandingReferences((current) => [...current, reference]);
    setMessage({ kind: "ok", text: `기립 위치 ${standingReferences.length + 1} 기록됨` });
  };

  const buildPayload = (): PostureCalibrationPayload => {
    if (!frameSize) throw new Error("카메라 영상 해상도를 아직 확인하지 못했습니다");
    if (floorPoints.length !== 4) throw new Error("영상에서 바닥 기준점 A·B·C·D를 찍으세요");
    const floorWorldPoints = floorWorldPointsFromAnchorDistances(
      floorPoints,
      numericAnchorDistances(anchorDistanceInputs),
    );
    if (standingReferences.length < 3) throw new Error("같은 사람이 서로 다른 세 위치에 섰을 때 각각 클릭하세요");
    return {
      frame_width: frameSize.width,
      frame_height: frameSize.height,
      floor_image_points: floorPoints,
      floor_world_points: floorWorldPoints,
      standing_references: standingReferences,
    };
  };

  const save = async () => {
    if (saving) return;
    let payload: PostureCalibrationPayload;
    try {
      payload = buildPayload();
    } catch (error) {
      setError(error instanceof Error ? error.message : "보정값을 확인하세요");
      return;
    }
    setSaving(true);
    try {
      const response = await fetch(`${apiBase()}/api/ipcams/${cameraId}/calibration`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await responseDetail(response, "보정값 저장에 실패했습니다"));
      const state = await response.json() as CalibrationState;
      setEnabled(state.enabled);
      setMessage({ kind: "ok", text: "저장됨 — 다음 추론 프레임부터 적용됩니다" });
    } catch (error) {
      setError(error instanceof Error ? error.message : "보정값 저장에 실패했습니다");
    } finally {
      setSaving(false);
    }
  };

  const clear = async () => {
    if (!enabled || saving || !window.confirm(`${cameraName} 자세 보정을 초기화할까요?`)) return;
    setSaving(true);
    try {
      const response = await fetch(`${apiBase()}/api/ipcams/${cameraId}/calibration`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error(await responseDetail(response, "보정 초기화에 실패했습니다"));
      setEnabled(false);
      setFrameSize(videoSize);
      setFloorPoints([]);
      setAnchorDistanceInputs(EMPTY_ANCHOR_DISTANCES);
      setStandingReferences([]);
      setStep("floor");
      setMessage({ kind: "ok", text: "초기화됨 — 기존 keypoint 판정으로 복귀했습니다" });
    } catch (error) {
      setError(error instanceof Error ? error.message : "보정 초기화에 실패했습니다");
    } finally {
      setSaving(false);
    }
  };

  const polygonPoints = floorHull.map((point) => point.join(",")).join(" ");
  const nextFloorPoint = FLOOR_POINT_LABELS[floorPoints.length];

  return (
    <Modal open={open} onClose={onClose} title={`바닥·원근 보정 · ${cameraName}`} maxWidth={1120}>
      <div style={styles.root}>
        <div style={styles.topline}>
          <div style={styles.status}>
            <span style={{ ...styles.dot, background: enabled ? "#3fb950" : "#8b95a5" }} />
            {enabled ? "보정 적용 중" : "보정 미설정"}
          </div>
          <div style={styles.steps}>
            <button style={{ ...styles.stepButton, ...(step === "floor" ? styles.stepButtonActive : {}) }} onClick={() => setStep("floor")}>
              1. 바닥 기준점 {floorPoints.length}/4
            </button>
            <button
              style={{ ...styles.stepButton, ...(step === "standing" ? styles.stepButtonActive : {}), ...(computedFloorWorldPoints ? {} : styles.disabled) }}
              disabled={!computedFloorWorldPoints}
              onClick={() => setStep("standing")}
            >
              2. 기립 기준 {standingReferences.length}/3+
            </button>
          </div>
        </div>

        <div style={styles.workspace}>
          <div style={styles.videoColumn}>
            <div style={styles.videoStage}>
              <WhepPlayer streamKey={streamKey} videoRef={videoRef} />
              <KeypointOverlay
                videoRef={videoRef}
                detections={inferenceActive ? items : []}
                frameW={frameW}
                frameH={frameH}
              />
              {frameSize && (
                <svg
                  aria-label="카메라 보정 좌표 선택 화면"
                  viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
                  preserveAspectRatio="xMidYMid meet"
                  onClick={handleVideoClick}
                  style={styles.clickOverlay}
                >
                  {floorHull.length >= 3 && (
                    <polygon points={polygonPoints} fill="rgba(68,147,248,0.14)" stroke="#4493f8" strokeWidth={Math.max(2, frameSize.width / 700)} />
                  )}
                  {MEASUREMENT_EDGES.map((edge) => {
                    const first = floorPoints[edge.first];
                    const second = floorPoints[edge.second];
                    if (!first || !second) return null;
                    const fontSize = Math.max(13, frameSize.width / 90);
                    return (
                      <g key={edge.label}>
                        <line x1={first[0]} y1={first[1]} x2={second[0]} y2={second[1]} stroke="#8fc1ff" strokeWidth={Math.max(1.5, frameSize.width / 950)} strokeDasharray="8 6" />
                        <text x={(first[0] + second[0]) / 2} y={(first[1] + second[1]) / 2 - 5} fill="#fff" fontSize={fontSize} fontWeight={800} textAnchor="middle" paintOrder="stroke" stroke="#0a0d12" strokeWidth={4}>
                          {edge.label}
                        </text>
                      </g>
                    );
                  })}
                  {floorPoints.map((point, index) => (
                    <g key={`floor-${index}`}>
                      <circle cx={point[0]} cy={point[1]} r={Math.max(9, frameSize.width / 130)} fill="#4493f8" stroke="#fff" strokeWidth={2} />
                      <text x={point[0]} y={point[1]} fill="#fff" fontSize={Math.max(13, frameSize.width / 85)} fontWeight={800} textAnchor="middle" dominantBaseline="central">
                        {String.fromCharCode(65 + index)}
                      </text>
                    </g>
                  ))}
                  {standingReferences.map((reference, index) => (
                    <g key={`standing-${index}`}>
                      <circle cx={reference.foot_px[0]} cy={reference.foot_px[1]} r={Math.max(8, frameSize.width / 150)} fill="#3fb950" stroke="#fff" strokeWidth={2} />
                      <text x={reference.foot_px[0]} y={reference.foot_px[1] - Math.max(14, frameSize.width / 80)} fill="#fff" fontSize={Math.max(13, frameSize.width / 90)} fontWeight={800} textAnchor="middle">
                        S{index + 1}
                      </text>
                    </g>
                  ))}
                </svg>
              )}
              {!videoSize && <div style={styles.videoNotice}>카메라 영상 연결 중…</div>}
            </div>
            <div style={styles.videoMeta}>
              {frameSize ? `보정 좌표 ${frameSize.width}×${frameSize.height}` : "영상 해상도 확인 중"}
              <span>사람 검출 {inferenceActive ? `${items.length}명` : "꺼짐"}</span>
            </div>
          </div>

          <aside style={styles.guide}>
            {loading ? <div style={styles.loading}>보정값 불러오는 중…</div> : step === "floor" ? (
              <>
                <div>
                  <h3 style={styles.heading}>임의의 바닥 기준점 4개 클릭</h3>
                  <p style={styles.help}>
                    네 점은 직사각형일 필요가 없습니다. 가구가 사이를 가려도 괜찮지만, A·B·C·D 자체는 모두 같은 실제 바닥면에 있어야 합니다.
                  </p>
                </div>
                <ol style={styles.pointList}>
                  {FLOOR_POINT_LABELS.map((label, index) => (
                    <li key={label} style={{ color: index < floorPoints.length ? "#3fb950" : index === floorPoints.length ? "#e6edf3" : "#657080" }}>
                      {index < floorPoints.length ? "✓" : index + 1}. {label}
                    </li>
                  ))}
                </ol>
                <div>
                  <div style={styles.measureTitle}>줄자·레이저로 잰 기준점 사이 거리</div>
                  <div style={styles.measureGrid}>
                    {DISTANCE_FIELDS.map((field) => {
                      const available = floorPoints.length >= field.requiredPoints;
                      return (
                        <label key={field.key} style={styles.field}>
                          <span style={styles.label}>{field.label} 실제 거리 (m)</span>
                          <input
                            style={{ ...styles.input, ...(available ? {} : styles.disabledInput) }}
                            inputMode="decimal"
                            disabled={!available}
                            value={anchorDistanceInputs[field.key]}
                            onChange={(event) => {
                              setAnchorDistanceInputs((current) => ({ ...current, [field.key]: event.target.value }));
                              setMessage(null);
                            }}
                            placeholder="0.00"
                          />
                        </label>
                      );
                    })}
                  </div>
                </div>
                <div style={styles.inlineActions}>
                  <button style={styles.secondaryButton} onClick={() => { setFloorPoints([]); setAnchorDistanceInputs(EMPTY_ANCHOR_DISTANCES); setStandingReferences([]); setMessage(null); }}>다시 찍기</button>
                  <button
                    style={{ ...styles.primaryButton, ...(computedFloorWorldPoints ? {} : styles.disabled) }}
                    disabled={!computedFloorWorldPoints}
                    onClick={() => setStep("standing")}
                  >
                    기립 기준으로
                  </button>
                </div>
                {nextFloorPoint ? (
                  <div style={styles.tip}>다음 클릭: <strong>{nextFloorPoint}</strong></div>
                ) : floorGeometryError ? (
                  <div style={styles.errorTip}>{floorGeometryError}</div>
                ) : !computedFloorWorldPoints ? (
                  <div style={styles.tip}>AB·AC·BC·AD·BD 거리를 모두 입력하면 삼각측량 결과를 확인합니다.</div>
                ) : (
                  <div style={styles.successTip}>거리 삼각측량 완료 · 기립 기준 단계로 이동할 수 있습니다.</div>
                )}
              </>
            ) : (
              <>
                <div>
                  <h3 style={styles.heading}>같은 사람을 세 위치에서 클릭</h3>
                  <p style={styles.help}>
                    사람이 가까운 곳·중간·먼 곳으로 이동해 똑바로 설 때마다 영상 속 몸을 한 번 클릭하세요. 발 위치와 keypoint 길이는 자동으로 기록됩니다.
                  </p>
                </div>
                <label style={styles.field}>
                  <span style={styles.label}>기준 사람 실제 키 (m)</span>
                  <input style={styles.input} inputMode="decimal" value={personHeightM} onChange={(event) => { setPersonHeightM(event.target.value); setMessage(null); }} placeholder="1.70" />
                </label>
                {!inferenceActive && (
                  <div style={styles.inferenceBox}>
                    <span>{hasModels ? "사람을 선택하려면 키포인트 추론이 필요합니다." : "먼저 이 카메라에 pose 모델을 선택하세요."}</span>
                    {hasModels && <button style={styles.primaryButton} onClick={onEnableInference}>추론 켜기</button>}
                  </div>
                )}
                {inferenceActive && items.length === 0 && <div style={styles.tip}>키포인트를 기다리는 중입니다. 사람이 전신으로 보이게 서 주세요.</div>}
                <div style={styles.sampleList}>
                  {standingReferences.length === 0 ? (
                    <div style={styles.emptySamples}>아직 기록된 기립 위치가 없습니다</div>
                  ) : standingReferences.map((reference, index) => (
                    <div key={`${reference.foot_px.join("-")}-${index}`} style={styles.sampleRow}>
                      <span><strong>S{index + 1}</strong> · 발 ({reference.foot_px[0].toFixed(0)}, {reference.foot_px[1].toFixed(0)})</span>
                      <button
                        aria-label={`기립 위치 ${index + 1} 삭제`}
                        style={styles.removeButton}
                        onClick={() => setStandingReferences((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                      >
                        삭제
                      </button>
                    </div>
                  ))}
                </div>
                <div style={styles.tip}>
                  {standingReferences.length < 3 ? `서로 다른 위치 ${3 - standingReferences.length}곳을 더 기록하세요.` : "필수 3곳 기록 완료 — 필요하면 위치를 더 추가할 수 있습니다."}
                </div>
              </>
            )}
          </aside>
        </div>

        {message && (
          <div role="status" style={{ ...styles.message, color: message.kind === "ok" ? "#3fb950" : "#e5484d" }}>
            {message.text}
          </div>
        )}
        <div style={styles.actions}>
          <button style={{ ...styles.secondaryButton, ...(enabled && !saving ? {} : styles.disabled) }} disabled={!enabled || saving} onClick={clear}>초기화</button>
          <button style={styles.secondaryButton} disabled={saving} onClick={onClose}>닫기</button>
          <button style={{ ...styles.primaryButton, ...(loading || saving ? styles.disabled : {}) }} disabled={loading || saving} onClick={save}>
            {saving ? "저장 중…" : "검증 후 저장"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: { display: "flex", flexDirection: "column", gap: "0.9rem" },
  topline: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.8rem", flexWrap: "wrap" },
  status: { display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.82rem", fontWeight: 700 },
  dot: { width: 8, height: 8, borderRadius: "50%" },
  steps: { display: "flex", gap: "0.4rem" },
  stepButton: { border: "1px solid #2a2f3a", background: "#11151c", color: "#8b95a5", borderRadius: 999, padding: "0.42rem 0.72rem", fontSize: "0.75rem", fontWeight: 700, cursor: "pointer" },
  stepButtonActive: { background: "rgba(229,72,77,0.14)", borderColor: "#e5484d", color: "#fff" },
  workspace: { display: "grid", gridTemplateColumns: "minmax(0, 1.8fr) minmax(280px, 0.8fr)", gap: "0.9rem", alignItems: "start" },
  videoColumn: { minWidth: 0 },
  videoStage: { position: "relative", aspectRatio: "16 / 9", overflow: "hidden", borderRadius: 10, border: "1px solid #2a2f3a", background: "#05070a" },
  clickOverlay: { position: "absolute", inset: 0, width: "100%", height: "100%", cursor: "crosshair", zIndex: 3 },
  videoNotice: { position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#8b95a5", background: "rgba(5,7,10,0.72)", fontSize: "0.82rem", pointerEvents: "none", zIndex: 4 },
  videoMeta: { display: "flex", justifyContent: "space-between", gap: "1rem", paddingTop: "0.4rem", color: "#8b95a5", fontSize: "0.7rem" },
  guide: { minHeight: 330, display: "flex", flexDirection: "column", gap: "0.8rem", padding: "0.9rem", border: "1px solid #2a2f3a", borderRadius: 10, background: "#11151c" },
  heading: { margin: "0 0 0.35rem", fontSize: "0.95rem" },
  help: { margin: 0, color: "#8b95a5", fontSize: "0.76rem", lineHeight: 1.55 },
  loading: { color: "#8b95a5", padding: "2rem 0", textAlign: "center" },
  pointList: { margin: 0, paddingLeft: "1.2rem", display: "flex", flexDirection: "column", gap: "0.38rem", fontSize: "0.78rem", fontWeight: 650 },
  measureTitle: { marginBottom: "0.45rem", color: "#c9d1d9", fontSize: "0.72rem", fontWeight: 700 },
  measureGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.55rem" },
  field: { display: "flex", flexDirection: "column", gap: "0.35rem" },
  label: { color: "#c9d1d9", fontSize: "0.72rem", fontWeight: 600 },
  input: { width: "100%", boxSizing: "border-box", background: "#0d1117", color: "#e6edf3", border: "1px solid #2a2f3a", borderRadius: 7, padding: "0.55rem 0.65rem", outline: "none" },
  disabledInput: { opacity: 0.38, cursor: "default" },
  inlineActions: { display: "flex", justifyContent: "space-between", gap: "0.5rem" },
  inferenceBox: { display: "flex", flexDirection: "column", gap: "0.55rem", border: "1px solid #523238", borderRadius: 8, padding: "0.65rem", color: "#c9d1d9", fontSize: "0.74rem" },
  sampleList: { display: "flex", flexDirection: "column", gap: "0.35rem", maxHeight: 135, overflowY: "auto" },
  sampleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem", padding: "0.45rem 0.55rem", borderRadius: 7, background: "#161b24", color: "#c9d1d9", fontSize: "0.7rem" },
  emptySamples: { padding: "0.8rem", textAlign: "center", color: "#657080", fontSize: "0.72rem", border: "1px dashed #2a2f3a", borderRadius: 7 },
  removeButton: { border: 0, background: "transparent", color: "#e5484d", fontSize: "0.68rem", cursor: "pointer" },
  tip: { padding: "0.55rem 0.65rem", borderRadius: 7, background: "rgba(68,147,248,0.09)", color: "#a9c7f6", fontSize: "0.72rem", lineHeight: 1.45 },
  successTip: { padding: "0.55rem 0.65rem", borderRadius: 7, background: "rgba(63,185,80,0.1)", color: "#71d17f", fontSize: "0.72rem", lineHeight: 1.45 },
  errorTip: { padding: "0.55rem 0.65rem", borderRadius: 7, background: "rgba(229,72,77,0.1)", color: "#ff7b81", fontSize: "0.72rem", lineHeight: 1.45 },
  message: { fontSize: "0.78rem" },
  actions: { display: "flex", justifyContent: "flex-end", gap: "0.55rem", paddingTop: "0.1rem" },
  secondaryButton: { border: "1px solid #2a2f3a", background: "#161b24", color: "#e6edf3", borderRadius: 7, padding: "0.5rem 0.8rem", cursor: "pointer" },
  primaryButton: { border: "none", background: "#e5484d", color: "white", borderRadius: 7, padding: "0.5rem 0.9rem", fontWeight: 700, cursor: "pointer" },
  disabled: { opacity: 0.45, cursor: "default" },
};

export default PostureCalibrationModal;
