import { useEffect, useRef } from "react";
import type { Cam } from "../pages/CamerasPage";
import WhepPlayer from "./WhepPlayer";
import KeypointOverlay from "./KeypointOverlay";
import { useDetectionWs } from "../hooks/useDetectionWs";
import { isSyncedPoseStream } from "../hooks/useApi";

// 사용자 오버라이드(게이트2) — 1줄 최대 4칸, 4 채워지면 다음 줄로 wrap.
function getGridColumns(count: number): number {
  return Math.min(Math.max(count, 1), 4);
}

interface Props {
  cams: Cam[];
  onFps?: (streamKey: string, fps: number) => void;
  // 카메라별 detection 메시지 도착률(det fps) 를 상위(CamerasPage)로 보고.
  onDetFps?: (streamKey: string, fps: number) => void;
  // 카메라별 추론 실활성 여부(추론 ON && 모델≥1) — 이 플래그가 검출 WS/캡처와 bbox 표시를
  //   둘 다 게이트한다: OFF 면 WS 를 안 열어 backend detection 캡처 미기동 + 스켈레톤 숨김
  //   (아래 GridCell 참고, codex #4). ON 전환 시 WS 개방→start_capture(RTSP cold open).
  inferenceActive: Record<string, boolean>;
  // 카메라별 remount epoch — CamerasPage 가 RTSP url 변경 성공 시에만 bump. GridCell key 에 섞어
  //   셀을 remount(WHEP 새 소스 재연결)한다. 마스킹 url 은 편집마다 문자열이 같아 rtsp_url-key 로는
  //   비번-only 편집을 remount 못 했다 → epoch 게이트가 정확히 변경분만 재연결.
  epochs?: Record<number, number>;
}

function SyncedPoseCell({
  cam,
  onFps,
  onDetFps,
}: {
  cam: Cam;
  onFps?: (streamKey: string, fps: number) => void;
  onDetFps?: (streamKey: string, fps: number) => void;
}) {
  const { annotatedFrame, detFps } = useDetectionWs(cam.stream_key, true);

  useEffect(() => {
    onDetFps?.(cam.stream_key, detFps);
    onFps?.(cam.stream_key, detFps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detFps, cam.stream_key]);

  return (
    <div className="grid-cell grid-cell-sync">
      {annotatedFrame ? (
        <img className="grid-cell-video grid-cell-sync-frame" src={annotatedFrame} alt="" />
      ) : (
        <span className="grid-cell-nosignal">동기화 대기</span>
      )}
    </div>
  );
}

// 셀 = WhepPlayer(WebRTC <video>) + 그 위 KeypointOverlay(canvas). .grid-cell 이 relative 컨테이너.
function OverlayCell({
  cam,
  onFps,
  onDetFps,
  inferenceOn,
}: {
  cam: Cam;
  onFps?: (streamKey: string, fps: number) => void;
  onDetFps?: (streamKey: string, fps: number) => void;
  inferenceOn: boolean;
}) {
  // WhepPlayer 의 <video> 를 KeypointOverlay 와 공유 — 같은 박스에 canvas 를 겹치기 위해.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // 검출 WS 는 추론 ON(inferenceOn) 일 때만 연결한다. WS connect==backend start_capture,
  //   disconnect==stop_capture (backend ipcam.py) 이므로, OFF 면 WS 를 열지 않아 백엔드
  //   detection 캡처(별도 RTSP 디코드)가 아예 기동되지 않는다 — spec: OFF = WHEP raw 영상만.
  //   ON 으로 토글하면 그때 WS 가 열려 start_capture 가 RTSP 를 cold open(카메라당 수 초)
  //   한 뒤 bbox 가 뜬다 — 수용한 트레이드오프다. 이전의 always-warm 방식은 최대 16대 카메라에서
  //   OFF 상태에서도 CPU/네트워크를 낭비했다(codex #4).
  const { items, frameW, frameH, captureTs, detFps } = useDetectionWs(cam.stream_key, inferenceOn);
  // det fps 를 상위로 보고 — 렌더 중 setState 금지이므로 effect 에서.
  useEffect(() => {
    onDetFps?.(cam.stream_key, detFps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detFps, cam.stream_key]);
  return (
    <div className="grid-cell">
      <WhepPlayer
        streamKey={cam.stream_key}
        videoRef={videoRef}
        onFps={(fps) => onFps?.(cam.stream_key, fps)}
      />
      <KeypointOverlay
        videoRef={videoRef}
        detections={inferenceOn ? items : []}
        captureTs={captureTs}
        frameW={frameW}
        frameH={frameH}
      />
    </div>
  );
}

function GridCell(props: {
  cam: Cam;
  onFps?: (streamKey: string, fps: number) => void;
  onDetFps?: (streamKey: string, fps: number) => void;
  inferenceOn: boolean;
}) {
  if (props.inferenceOn && (props.cam.sync_pose === true || isSyncedPoseStream(props.cam.stream_key))) {
    return <SyncedPoseCell cam={props.cam} onFps={props.onFps} onDetFps={props.onDetFps} />;
  }
  return <OverlayCell {...props} />;
}

export default function CameraGrid({ cams, onFps, onDetFps, inferenceActive, epochs }: Props) {
  if (cams.length === 0) {
    return <p className="grid-empty">등록된 카메라가 없습니다.</p>;
  }

  const columns = getGridColumns(cams.length);

  return (
    <div className="grid" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
      {cams.map((cam) => (
        // key 에 playerEpoch 포함 — RTSP url 변경 성공 시에만 CamerasPage 가 bump 해 셀을 remount,
        // WHEP 를 새 소스로 재연결한다. 마스킹 url(:***@)은 편집마다 문자열이 같아 rtsp_url-key 로는
        // 비번-only 편집을 remount 못 했고 이름-only 편집은 불필요 remount 했다 → epoch 게이트로 정확화.
        <GridCell
          key={`${cam.id}-${epochs?.[cam.id] ?? 0}`}
          cam={cam}
          onFps={onFps}
          onDetFps={onDetFps}
          inferenceOn={inferenceActive[cam.stream_key] ?? false}
        />
      ))}
    </div>
  );
}
