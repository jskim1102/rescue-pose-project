import { useEffect, useRef, useState } from "react";
import { whepBase, whepAuthHeaders } from "../hooks/useApi";

interface Props {
  streamKey: string;
  onFps?: (fps: number) => void;
  // 부모가 <video> 를 공유받기 위한 ref(선택) — BboxOverlay 가 이 위에 canvas 를 겹친다.
  // 주면 그걸 video 에 붙이고(동시에 WebRTC 도 사용), 없으면 내부 ref.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

// mediamtx WHEP 플레이어 — native RTCPeerConnection. 무거운 lib 없이 ~40줄.
// createOffer → POST SDP → setRemoteDescription(answer) → ontrack → <video>.
export default function WhepPlayer({ streamKey, onFps, videoRef: externalRef }: Props) {
  const internalRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalRef ?? internalRef;
  const [failed, setFailed] = useState(false);
  // onFps 를 ref 로 잡아 effect dep 에서 뺀다 — inline 콜백이 매 렌더 새로 생겨도
  // WebRTC 가 재연결되지 않도록(dep=[streamKey] 만 유지).
  const onFpsRef = useRef(onFps);
  onFpsRef.current = onFps;

  useEffect(() => {
    const pc = new RTCPeerConnection();
    let aborted = false;

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    pc.ontrack = (ev) => {
      if (videoRef.current) videoRef.current.srcObject = ev.streams[0];
    };

    // 실측 FPS — mediamtx API 엔 FPS 가 없고(백엔드는 프레임 안 만짐) WebRTC 로
    // 디코딩되는 실제 프레임레이트를 잰다: inbound-rtp(video) framesDecoded 증분 / 시간.
    let lastFrames = 0;
    let lastTs = 0;
    const fpsTimer = window.setInterval(async () => {
      const stats = await pc.getStats();
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
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const resp = await fetch(`${whepBase()}/${streamKey}/whep`, {
          method: "POST",
          headers: { "Content-Type": "application/sdp", ...whepAuthHeaders() },
          body: offer.sdp,
        });
        if (!resp.ok) throw new Error(`WHEP ${resp.status}`);

        const answer = await resp.text();
        if (aborted) return;
        await pc.setRemoteDescription({ type: "answer", sdp: answer });
      } catch {
        if (!aborted) setFailed(true);
      }
    })();

    return () => {
      aborted = true;
      window.clearInterval(fpsTimer);
      pc.close();
    };
  }, [streamKey]);

  if (failed) {
    return <span className="grid-cell-nosignal">연결 실패</span>;
  }

  return <video ref={videoRef} className="grid-cell-video" autoPlay muted playsInline />;
}
