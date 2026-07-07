import { useEffect, useRef, useState } from "react";
import { apiBase } from "./useApi";
import type { KeypointPerson, Posture } from "../components/KeypointOverlay";

// 계약: item{keypoints:[[x,y,c]×17], model, posture?:'standing'|'sitting'|'lying'}.
const VALID_POSTURES: readonly Posture[] = ["standing", "sitting", "lying"];

// WS item → KeypointPerson. posture 는 계약 union 으로 검증(그 외/부재 = undefined).
// rescueNeeded/lyingSec(phase4 rescue seam)도 통과 — 대시보드 UI 가 소비.
function parsePerson(raw: {
  keypoints?: [number, number, number][];
  model?: string;
  posture?: string;
  rescueNeeded?: boolean;
  lyingSec?: number;
}): KeypointPerson {
  const posture = VALID_POSTURES.includes(raw?.posture as Posture)
    ? (raw.posture as Posture)
    : undefined;
  return {
    keypoints: raw?.keypoints ?? [],
    model: raw?.model ?? "",
    posture,
    rescueNeeded: raw?.rescueNeeded === true,
    lyingSec: typeof raw?.lyingSec === "number" ? raw.lyingSec : undefined,
  };
}

/**
 * 카메라별 detection 좌표 WS 구독 (좌표 전용 — 하이브리드).
 *
 * deepeye 원본 WS 는 raw JPEG(binary) + detections(text) 둘 다 받았지만, 여기선 영상이
 * WHEP `<video>` 로 따로 오므로 이 WS 는 **text(detections JSON) 만** 파싱한다(binary 무시).
 *
 *   ws://<host>:<VITE_API_PORT>/api/ipcams/{streamKey}/ws
 *   ← { type:"detections", items:[...], frame:{w,h} }
 *
 * - `active=false` 면 연결하지 않고(백엔드 캡처 안 띄움) items 를 즉시 비운다.
 * - `active=true` 면 연결, 끊기면 2s 후 자동 reconnect.
 * - frame{w,h} = YOLO 가 본 추론 캡처 프레임 치수(SEAM 좌표 스케일용). 없으면 0(=identity).
 */

const RECONNECT_DELAY = 2000;
const DET_FPS_WINDOW_MS = 2000; // det fps 측정 윈도우(~2초).

export interface DetectionStream {
  items: KeypointPerson[];
  frameW: number;
  frameH: number;
  captureTs: number | null;
  detFps: number;
  annotatedFrame: string | null;
}

export function useDetectionWs(
  streamKey: string,
  active: boolean,
  onEvents?: (reasons: string[]) => void,
): DetectionStream {
  // 콜백을 ref 로 잡아 effect 재구독 없이 최신 핸들러 호출(rescue 종료 이벤트 전달용).
  const onEventsRef = useRef(onEvents);
  onEventsRef.current = onEvents;
  const [items, setItems] = useState<KeypointPerson[]>([]);
  const [frame, setFrame] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  const [captureTs, setCaptureTs] = useState<number | null>(null);
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null);
  // detection 메시지 도착률(det fps) — 추론 ON 일 때만 백엔드가 메시지를 보냄(OFF=0).
  const [detFps, setDetFps] = useState(0);
  // 최근 도착 타임스탬프(performance.now()) 윈도우 — 렌더 무관하게 ref 로 관리.
  const detArrivalsRef = useRef<number[]>([]);

  useEffect(() => {
    // 추론 OFF / 미연결 — WS 안 열고(백엔드 capture 미기동) 잔존 detection·측정치 즉시 정리.
    if (!active) {
      setItems([]);
      setCaptureTs(null);
      setAnnotatedFrame(null);
      detArrivalsRef.current = [];
      setDetFps(0);
      return;
    }

    let unmounted = false;
    let reconnectTimer: number | null = null;
    let ws: WebSocket | null = null;

    // det fps 갱신: 윈도우 밖 타임스탬프를 버리고 (윈도우 내 개수)/(윈도우초) 계산.
    const detFpsTimer = window.setInterval(() => {
      if (unmounted) return;
      const cutoff = performance.now() - DET_FPS_WINDOW_MS;
      const arr = detArrivalsRef.current.filter((t) => t >= cutoff);
      detArrivalsRef.current = arr;
      setDetFps(arr.length / (DET_FPS_WINDOW_MS / 1000));
      // detection 침묵 감지 — WS 는 열려 있으나 윈도우(2s) 내 메시지가 0건이면(추론 OFF·RTSP drop
      // 등으로 백엔드가 push 중단) 잔존 items 를 정리한다. 안 그러면 마지막 detection 이 얼어붙어
      // stale rescue 배너·유령 counts 가 안 사라진다(onclose 만으론 open-socket 침묵 미커버).
      // functional update — 이미 비어 있으면 no-op(startup·정상 빈상태에서 불필요 re-render 방지).
      if (arr.length === 0) {
        setItems((prev) => (prev.length ? [] : prev));
        setFrame((prev) => (prev.w || prev.h ? { w: 0, h: 0 } : prev));
        setCaptureTs((prev) => (prev === null ? prev : null));
        setAnnotatedFrame((prev) => (prev === null ? prev : null));
      }
    }, 1000);
    // ws URL = apiBase() 의 http→ws 치환 (동일 host:VITE_API_PORT).
    const wsUrl = `${apiBase().replace(/^http/, "ws")}/api/ipcams/${streamKey}/ws`;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(wsUrl);

      ws.onmessage = (event: MessageEvent) => {
        if (unmounted) return;
        // 슬림 — text(detections JSON) 만. binary 프레임은 오지 않음(WHEP 가 영상 담당).
        if (typeof event.data !== "string") return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "detections") {
            // 도착 시각 기록 → det fps 측정(추론 ON 일 때만 메시지가 옴).
            detArrivalsRef.current.push(performance.now());
            // posture 포함 items 파싱 — parsePerson 이 posture 를 계약 union 으로 검증해 통과.
            setItems((msg.items ?? []).map(parsePerson) as KeypointPerson[]);
            setCaptureTs(typeof msg.timestamp === "number" ? msg.timestamp : null);
            // frame{w,h} 동봉 시 갱신(SEAM). 없으면 0 유지 → KeypointOverlay identity.
            if (msg.frame) {
              setFrame({ w: msg.frame.w ?? 0, h: msg.frame.h ?? 0 });
            }
            setAnnotatedFrame(
              typeof msg.annotatedFrame === "string" && msg.annotatedFrame.startsWith("data:image/")
                ? msg.annotatedFrame
                : null,
            );
            // rescue 종료 이벤트(recovered/lost) — 동봉 시 부모로 전달(이벤트 로그).
            if (Array.isArray(msg.rescueEvents) && msg.rescueEvents.length) {
              onEventsRef.current?.(msg.rescueEvents as string[]);
            }
          }
        } catch {
          /* malformed — 무시 */
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        // 연결 끊김 — 잔존 detection 이 얼어붙어(스켈레톤 정지 + posture 집계 유령) 표시되는 걸
        // 즉시 막는다(rescue 앱 false-state 방지). 빈 items 로 KeypointOverlay interp 버퍼도 자연 소멸.
        setItems([]);
        setFrame({ w: 0, h: 0 });
        setCaptureTs(null);
        setAnnotatedFrame(null);
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    // streamKey 변경으로 effect 재실행 시 이전 스트림의 items/frame 을 이어받지 않도록 초기화
    // (카메라 전환 순간 이전 카메라 사람/자세가 새 슬롯에 잠깐 보이는 것 방지). 첫 메시지가 채운다.
    setItems([]);
    setFrame({ w: 0, h: 0 });
    setCaptureTs(null);
    setAnnotatedFrame(null);

    connect();

    return () => {
      unmounted = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      clearInterval(detFpsTimer);
      detArrivalsRef.current = [];
      setDetFps(0);
      setCaptureTs(null);
      setAnnotatedFrame(null);
      if (ws) ws.close();
    };
  }, [streamKey, active]);

  return { items, frameW: frame.w, frameH: frame.h, captureTs, detFps, annotatedFrame };
}
