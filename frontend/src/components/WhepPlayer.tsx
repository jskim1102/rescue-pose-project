import { useEffect, useRef, useState } from "react";
import { whepBase, whepAuthHeaders } from "../hooks/useApi";

interface Props {
  streamKey: string;
  onFps?: (fps: number) => void;
  // 부모가 <video> 를 공유받기 위한 ref(선택) — KeypointOverlay 가 이 위에 canvas 를 겹친다.
  // 주면 그걸 video 에 붙이고(동시에 WebRTC 도 사용), 없으면 내부 ref.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

// mediamtx WHEP 플레이어 — native RTCPeerConnection.
// createOffer → POST SDP → setRemoteDescription(answer) → ontrack → <video>.
// WHEP 실패·연결 끊김 시 backoff 로 제한적 재시도(mediamtx 재시작/일시 404·503·네트워크 복원력).
export default function WhepPlayer({ streamKey, onFps, videoRef: externalRef }: Props) {
  const internalRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalRef ?? internalRef;
  const [failed, setFailed] = useState(false);
  // onFps 를 ref 로 잡아 effect dep 에서 뺀다 — inline 콜백이 매 렌더 새로 생겨도
  // WebRTC 가 재연결되지 않도록(dep=[streamKey] 만 유지).
  const onFpsRef = useRef(onFps);
  onFpsRef.current = onFps;

  useEffect(() => {
    let aborted = false;
    let pc: RTCPeerConnection | null = null;
    let fpsTimer: number | undefined;
    let retryTimer: number | undefined;
    let attempt = 0;
    const MAX_RETRIES = 5;

    function teardown() {
      if (fpsTimer !== undefined) {
        window.clearInterval(fpsTimer);
        fpsTimer = undefined;
      }
      if (pc) {
        pc.onconnectionstatechange = null;
        pc.ontrack = null;
        pc.close();
        pc = null;
      }
    }

    function onFailure() {
      if (aborted) return;
      teardown();
      if (attempt >= MAX_RETRIES) {
        setFailed(true);
        return;
      }
      const delay = Math.min(1000 * 2 ** attempt, 15000);
      attempt += 1;
      retryTimer = window.setTimeout(connect, delay);
    }

    function connect() {
      if (aborted) return;
      teardown();
      setFailed(false);

      const conn = new RTCPeerConnection();
      pc = conn;
      conn.addTransceiver("video", { direction: "recvonly" });

      conn.ontrack = (ev) => {
        if (videoRef.current) videoRef.current.srcObject = ev.streams[0];
      };

      conn.onconnectionstatechange = () => {
        if (aborted || conn !== pc) return;
        const state = conn.connectionState;
        if (state === "connected") attempt = 0;
        else if (state === "failed" || state === "disconnected") onFailure();
      };

      // 실측 FPS — mediamtx API 엔 FPS 가 없고 WebRTC 로 디코딩되는 실제 프레임레이트를 잰다.
      let lastFrames = 0;
      let lastTs = 0;
      fpsTimer = window.setInterval(async () => {
        const stats = await conn.getStats();
        stats.forEach((raw) => {
          const r = raw as RTCInboundRtpStreamStats & { mediaType?: string };
          const kind = r.mediaType ?? r.kind;
          if (r.type === "inbound-rtp" && kind === "video" && r.framesDecoded != null) {
            if (lastTs) {
              const dt = (r.timestamp - lastTs) / 1000;
              if (dt > 0) onFpsRef.current?.(Math.max(0, (r.framesDecoded - lastFrames) / dt));
            }
            lastFrames = r.framesDecoded;
            lastTs = r.timestamp;
          }
        });
      }, 1000);

      (async () => {
        try {
          const offer = await conn.createOffer();
          await conn.setLocalDescription(offer);

          const resp = await fetch(`${whepBase()}/${streamKey}/whep`, {
            method: "POST",
            headers: { "Content-Type": "application/sdp", ...whepAuthHeaders() },
            body: offer.sdp,
          });
          if (!resp.ok) throw new Error(`WHEP ${resp.status}`);

          const answer = await resp.text();
          if (aborted || conn !== pc) return;
          await conn.setRemoteDescription({ type: "answer", sdp: answer });
        } catch {
          if (!aborted && conn === pc) onFailure();
        }
      })();
    }

    connect();

    return () => {
      aborted = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      teardown();
    };
  }, [streamKey]);

  if (failed) {
    return <span className="grid-cell-nosignal">연결 실패</span>;
  }

  return <video ref={videoRef} className="grid-cell-video" autoPlay muted playsInline />;
}
