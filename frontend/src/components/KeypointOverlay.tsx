import { useEffect, useRef, useState } from "react";
import {
  KPT_COLOR_IDX,
  SKELETON_EDGES,
  kptColor,
  limbColor,
} from "../utils/colors";

/**
 * WHEP `<video>` 위에 절대 위치 `<canvas>` 로 **스켈레톤 keypoint** 오버레이. ← 하이브리드 SEAM
 *
 * 부모 BboxOverlay(rect draw)를 pose 로 스왑한 것. 영상(WHEP)과 추론 프레임(백엔드 별도 캡처)이
 * **서로 다른 한 장**이라 두 좌표계를 정합하는 SEAM 스케일 로직은 그대로 계승한다:
 *
 * - canvas internal width/height = 부모 `<video>` 의 natural 크기(video.videoWidth/Height)
 *   — `loadedmetadata` · `resize` 이벤트로 갱신(WHEP 는 srcObject 라 `load` 가 없음).
 * - keypoints 좌표는 추론 캡처 프레임(frameW×frameH) 좌표계 → `videoNatural / detFrame`
 *   스케일로 그린다. **두 해상도가 같으면 sx=sy=1 (identity), 다르면 자동 보정.**
 *   (frameW/H 미상=0 이면 identity 가정.)
 * - canvas 는 video 와 동일 박스에 절대배치(inset:0, 100%)하되 `object-fit:contain` 으로
 *   video 와 똑같이 레터/필러박싱 → 비 16:9 카메라(예 4:3)에서도 화면상 정렬.
 *
 * pose 는 단일 class(person)라 class filter/색상 override 가 없다 → settings prop 없음.
 * 사람 단위 conf 는 서버(워커)에서 이미 적용됨. per-keypoint viz 임계는 아래 상수.
 *
 * **보간/외삽 (CEO #193)**: WHEP 영상은 디스플레이 fps(30~60)로 재생되나 추론은 10fps
 * (INFERENCE_INTERVAL=0.1, 백엔드 유지)라, detections 바뀔 때만 그리면 스켈레톤이 ~10fps 로
 * 끊기며 ~100ms 뒤처진다. 해결: 최근 2개 추론결과(+backend frame ts)를 버퍼링해
 * `requestAnimationFrame` 루프로 매 디스플레이 프레임 재렌더 — 두 프레임의 속도로 현재 시각까지 keypoint 위치를 외삽
 * (지연 보상)한다. 외삽은 1 추론주기(~100ms)로 clamp(오버슛/지터 방지). 트랙 ID 가 없으므로
 * (YOLO detect) 프레임 간 사람을 centroid 거리로 greedy 매칭한 뒤 매칭쌍만 외삽하고, 매칭
 * 안 된 사람(신규 등장)은 ghost 외삽 없이 최신 위치로 스냅한다. 소멸한 사람은 최신 프레임에
 * 없으므로 자연히 안 그려진다(ghost 없음).
 */

// per-keypoint 표시 임계 — conf 미만 관절 점/엣지 미표시 (spec Unknowns: 0.5 시작, gate2 튜닝 knob).
const KPT_CONF_THRESHOLD = 0.5;

// 외삽 clamp — 최신 프레임 이후 이 시간(ms)까지만 속도 외삽 (오버슛/지터 방지).
const MAX_EXTRAP_MS = 120;
const MAX_TRUSTED_CAPTURE_AGE_MS = 10_000;

// 두 버퍼 프레임의 수신 간격이 이 값(ms) 미만이면 속도 계산을 신뢰하지 않고 스냅.
// StrictMode 이중호출/중복 push 로 dt≈0 이 되면 속도가 폭발하는 것을 방지.
const MIN_DT_MS = 20;

// cam1(NVR/WHEP overlay) 표시 전용 jitter filter.
// 작은 떨림은 고정하고, 실제 이동은 높은 alpha로 따라가 지연감을 만들지 않는다.
const JITTER_DEADBAND_PX = 2.5;
const JITTER_SOFTEN_PX = 18;
const JITTER_SNAP_PX = 70;
const JITTER_ALPHA_SMALL = 0.38;
const JITTER_ALPHA_LARGE = 0.78;

// 자세분류 결과 — backend _classify_posture(COCO17) → WS payload item.posture.
export type Posture = "standing" | "sitting" | "lying";

// posture 라벨 — 스켈레톤 상단에 자세(standing/sitting/lying) 배지. rescue 테마상
// lying=위험(red)/sitting=주의(amber)/standing=정상(green) 로 색을 매핑한다.
// phase3: 실 posture 를 WS payload(item.posture)로 배선 — person.posture 가 있을 때만
// 배지를 그린다(미상이면 라벨 없음, mock 폴백 없음).
const POSTURE_LABEL_COLOR: Record<Posture, string> = {
  lying: "#e5484d",
  sitting: "#e0a23b",
  standing: "#3fb950",
};

// 사람 1명 = 17 COCO keypoint, 각 점 = [x, y, conf] (추론 캡처 프레임 좌표계).
// posture = tunnel `_classify_posture` 가 WS payload(item.posture)로 채운다(phase3 실배선).
// rescue seam(phase4) — backend RescueTracker → WS payload. 대시보드 UI 만 소비(오버레이는 미사용).
export interface KeypointPerson {
  keypoints: [number, number, number][];
  model: string;
  posture?: Posture;
  rescueNeeded?: boolean;
  lyingSec?: number;
}

// 추론결과 1건 + 프레임 시각(performance.now 기준). rAF 루프가 두 개를 보고 외삽.
interface BufferedFrame {
  persons: KeypointPerson[];
  t: number;
}

interface SmoothedPerson {
  keypoints: [number, number, number][];
}

function captureTimestampToPerformance(captureTs?: number | null): number {
  const now = performance.now();
  if (typeof captureTs !== "number" || !Number.isFinite(captureTs) || captureTs <= 0) {
    return now;
  }
  const ageMs = Date.now() - captureTs * 1000;
  if (!Number.isFinite(ageMs) || ageMs < -100 || ageMs > MAX_TRUSTED_CAPTURE_AGE_MS) {
    return now;
  }
  return now - Math.max(0, ageMs);
}

// conf>0 keypoint 들의 평균 좌표(사람 centroid). 유효 점이 없으면 null.
function kpCentroid(p: KeypointPerson): { x: number; y: number } | null {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const [x, y, c] of p.keypoints) {
    if (c > 0) {
      sx += x;
      sy += y;
      n++;
    }
  }
  return n ? { x: sx / n, y: sy / n } : null;
}

/**
 * 프레임 간 사람 greedy 매칭 (트랙 ID 없음 → centroid 최근접). 반환: 최신[i] → 이전[j] | -1.
 * 전역 최근접(모든 쌍 거리 정렬 후 greedy 할당)이라 사람 목록 순서에 무관하게 안정적 —
 * per-person 순차 greedy 보다 다중 인물 joint 스왑(엉뚱한 사람끼리 섞임) 위험이 낮다.
 */
function matchPersons(
  older: KeypointPerson[],
  newer: KeypointPerson[],
  maxDist: number,
): number[] {
  const oc = older.map(kpCentroid);
  const nc = newer.map(kpCentroid);
  const out = new Array<number>(newer.length).fill(-1);

  const pairs: [number, number, number][] = []; // [dist, newerIdx, olderIdx]
  for (let i = 0; i < newer.length; i++) {
    const ni = nc[i];
    if (!ni) continue;
    for (let j = 0; j < older.length; j++) {
      const oj = oc[j];
      if (!oj) continue;
      const d = Math.hypot(ni.x - oj.x, ni.y - oj.y);
      if (d <= maxDist) pairs.push([d, i, j]);
    }
  }
  pairs.sort((a, b) => a[0] - b[0]);

  const usedNew = new Set<number>();
  const usedOld = new Set<number>();
  for (const [, i, j] of pairs) {
    if (usedNew.has(i) || usedOld.has(j)) continue;
    out[i] = j;
    usedNew.add(i);
    usedOld.add(j);
  }
  return out;
}

function smoothKeypoint(
  prev: [number, number, number] | undefined,
  next: [number, number, number],
): [number, number, number] {
  const [nx, ny, nc] = next;
  if (!prev || nc < KPT_CONF_THRESHOLD || prev[2] < KPT_CONF_THRESHOLD) return next;

  const dx = nx - prev[0];
  const dy = ny - prev[1];
  const dist = Math.hypot(dx, dy);
  if (dist <= JITTER_DEADBAND_PX) return [prev[0], prev[1], nc];
  if (dist >= JITTER_SNAP_PX) return next;

  const alpha = dist >= JITTER_SOFTEN_PX ? JITTER_ALPHA_LARGE : JITTER_ALPHA_SMALL;
  return [prev[0] + dx * alpha, prev[1] + dy * alpha, nc];
}

function smoothPose(
  prev: SmoothedPerson | undefined,
  nextKeypoints: [number, number, number][],
): SmoothedPerson {
  return {
    keypoints: nextKeypoints.map((kp, i) => smoothKeypoint(prev?.keypoints[i], kp)),
  };
}

interface Props {
  // 부모 WhepPlayer 의 <video> ref — 이 위에 절대배치 canvas 를 겹친다.
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections: KeypointPerson[];
  // backend frame timestamp(seconds). 없으면 WebSocket 수신 시각으로 fallback.
  captureTs?: number | null;
  // YOLO 가 본 추론 캡처 프레임 치수 (WS frame:{w,h}). 0 이면 스케일 = identity.
  frameW: number;
  frameH: number;
}

function KeypointOverlay({ videoRef, detections, captureTs, frameW, frameH }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  // 최근 2개 추론결과 링버퍼([이전, 최신]) — rAF 루프가 여기서 읽어 외삽.
  const bufferRef = useRef<BufferedFrame[]>([]);
  const smoothedRef = useRef<SmoothedPerson[]>([]);

  // video natural 크기 — loadedmetadata + resize 시 갱신.
  // WHEP 는 srcObject(MediaStream) 라 <img> 의 onload 가 없다. 트랙 해상도 확정/변경은
  // video 의 'resize'(intrinsic 크기 변동) + 'loadedmetadata' 로 통지된다.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => {
      if (video.videoWidth && video.videoHeight) {
        setSize((prev) => {
          if (prev?.w === video.videoWidth && prev?.h === video.videoHeight) return prev;
          return { w: video.videoWidth, h: video.videoHeight };
        });
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    update(); // 이미 메타데이터 로드됐을 수 있음
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [videoRef]);

  // 새 detections 도착 → 프레임 ts 와 함께 버퍼에 push(최근 2개만 유지). 그리지 않음.
  // useDetectionWs 가 메시지마다 새 배열을 주므로 ref 변경마다 1회 push.
  useEffect(() => {
    const buf = bufferRef.current;
    buf.push({ persons: detections, t: captureTimestampToPerformance(captureTs) });
    if (buf.length > 2) buf.shift();
    if (detections.length === 0) smoothedRef.current = [];
  }, [detections, captureTs]);

  // rAF 렌더 루프 — 매 디스플레이 프레임 버퍼에서 외삽 위치를 계산해 재그림.
  // size/frameW/frameH 변경 시 루프 재시작(클로저 갱신). 언마운트 시 취소.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // SEAM 좌표 스케일 — keypoint(추론 프레임 좌표계) → canvas internal(video natural).
    // 두 해상도 같으면 identity. (frameW/H=0 이면 identity 가정.)
    const sx = frameW > 0 ? size.w / frameW : 1;
    const sy = frameH > 0 ? size.h / frameH : 1;

    // 원본 좌표계 기준 적정 사이즈 (해상도 클수록 선 두께/점 반경 비율 보정)
    const scale = Math.max(1, Math.min(size.w, size.h) / 600);
    const lineWidth = Math.max(1.5, 2 * scale);
    const pointRadius = Math.max(2, 3 * scale);
    const fontPx = Math.max(11, Math.round(12 * scale)); // posture 라벨 폰트

    // centroid 매칭 최대 거리 — 프레임 대각선의 25%. frameW/H 미상이면 video natural 로 대용
    // (identity 가정과 일관). 100ms 새 사람 이동이 이보다 크면 매칭 안 하고 스냅(오매칭 방지).
    const effFrameW = frameW > 0 ? frameW : size.w;
    const effFrameH = frameH > 0 ? frameH : size.h;
    const maxMatchDist = 0.25 * Math.hypot(effFrameW, effFrameH);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const buf = bufferRef.current;
      if (buf.length === 0) return;

      const newest = buf[buf.length - 1];
      const older = buf.length >= 2 ? buf[buf.length - 2] : null;
      const dt = older ? newest.t - older.t : 0;
      const canExtrap = older !== null && dt >= MIN_DT_MS;

      // 최신 프레임 이후 경과 — 외삽량. [0, MAX_EXTRAP_MS] clamp.
      const extra = canExtrap
        ? Math.min(Math.max(performance.now() - newest.t, 0), MAX_EXTRAP_MS)
        : 0;

      // 프레임 간 사람 매칭 (외삽 가능할 때만).
      const match = canExtrap
        ? matchPersons(older!.persons, newest.persons, maxMatchDist)
        : null;
      const nextSmoothed: SmoothedPerson[] = [];

      ctx.lineWidth = lineWidth;

      for (let pi = 0; pi < newest.persons.length; pi++) {
        const person = newest.persons[pi];
        const kps = person.keypoints;
        if (!kps || kps.length < 17) continue;

        const oldIdx = match ? match[pi] : -1;
        const oldKps = oldIdx >= 0 ? older!.persons[oldIdx].keypoints : null;

        // 17점 각각 외삽/스냅. conf 는 항상 최신 프레임 값 사용.
        // 매칭 사람 & 양 프레임 conf 충분 → 속도(=Δ/dt) 외삽; 그 외(신규/저신뢰/외삽불가) → 스냅.
        const pts: [number, number, number][] = new Array(17);
        for (let i = 0; i < 17; i++) {
          const [nx, ny, nc] = kps[i];
          if (oldKps && canExtrap) {
            const [ox, oy, oc] = oldKps[i];
            if (oc >= KPT_CONF_THRESHOLD && nc >= KPT_CONF_THRESHOLD) {
              const vx = (nx - ox) / dt;
              const vy = (ny - oy) / dt;
              pts[i] = [nx + vx * extra, ny + vy * extra, nc];
              continue;
            }
          }
          pts[i] = [nx, ny, nc];
        }
        const smoothed = smoothPose(smoothedRef.current[pi], pts);
        nextSmoothed[pi] = smoothed;
        const drawPts = smoothed.keypoints;

        // 1) 스켈레톤 엣지 — 두 끝점 conf 가 모두 임계 이상일 때만 선(보간 좌표로).
        for (let e = 0; e < SKELETON_EDGES.length; e++) {
          const [a, b] = SKELETON_EDGES[e];
          const pa = drawPts[a];
          const pb = drawPts[b];
          if (pa[2] < KPT_CONF_THRESHOLD || pb[2] < KPT_CONF_THRESHOLD) continue;
          ctx.strokeStyle = limbColor(e);
          ctx.beginPath();
          ctx.moveTo(pa[0] * sx, pa[1] * sy);
          ctx.lineTo(pb[0] * sx, pb[1] * sy);
          ctx.stroke();
        }

        // 2) keypoint 점 — conf 임계 이상만 원(보간 좌표로).
        for (let i = 0; i < KPT_COLOR_IDX.length; i++) {
          const p = drawPts[i];
          if (!p || p[2] < KPT_CONF_THRESHOLD) continue;
          ctx.fillStyle = kptColor(i);
          ctx.beginPath();
          ctx.arc(p[0] * sx, p[1] * sy, pointRadius, 0, Math.PI * 2);
          ctx.fill();
        }

        // 3) posture 라벨 — 스켈레톤 상단(유효 점의 좌·상단)에 자세 배지.
        //    phase3: 실 posture(person.posture, WS payload) 있을 때만 렌더(mock 없음).
        const posture = person.posture;
        let minX = Infinity;
        let minY = Infinity;
        for (const p of drawPts) {
          if (!p || p[2] < KPT_CONF_THRESHOLD) continue;
          const cx = p[0] * sx;
          const cy = p[1] * sy;
          if (cx < minX) minX = cx;
          if (cy < minY) minY = cy;
        }
        if (posture && Number.isFinite(minX) && Number.isFinite(minY)) {
          const badge = POSTURE_LABEL_COLOR[posture] ?? "#8b95a5";
          const text = posture.toUpperCase();
          ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
          ctx.textBaseline = "top";
          const padX = 5;
          const padY = 3;
          const textW = ctx.measureText(text).width;
          const labelH = fontPx + padY * 2;
          const ly = Math.max(0, minY - labelH - 2);
          ctx.fillStyle = badge;
          ctx.fillRect(minX, ly, textW + padX * 2, labelH);
          ctx.fillStyle = "#0a0d12"; // 밝은 배지 위 다크 텍스트(고대비)
          ctx.fillText(text, minX + padX, ly + padY);
        }
      }
      smoothedRef.current = nextSmoothed;
    };

    let raf = requestAnimationFrame(function loop() {
      draw();
      raf = requestAnimationFrame(loop);
    });
    return () => cancelAnimationFrame(raf);
  }, [size, frameW, frameH]);

  // video 메타데이터가 아직이면(size 미상) 그릴 대상 없음 — 오버레이 생략.
  if (!size) return null;

  return (
    <canvas
      ref={canvasRef}
      width={size.w}
      height={size.h}
      style={canvasStyle}
    />
  );
}

const canvasStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  pointerEvents: "none",
};

export default KeypointOverlay;
