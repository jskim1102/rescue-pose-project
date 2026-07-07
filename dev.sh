#!/usr/bin/env bash
# 호스트 dev 러너 (하이브리드) — backend/frontend=native(venv uvicorn+vite),
#   mediamtx=docker. ./dev.sh up 이 셋 다 올리고 down 이 셋 다 내린다.
# mediamtx 는 image-only(bluenviron/mediamtx) — 호스트 바이너리는 없지만 docker 로 기동.
#   compose 가 .env offset 포트로 remap + ffmpeg 변형 제공 → 네이티브 포트수동 회피.
set -euo pipefail
cd "$(dirname "$0")"
set -a; [ -f .env ] && source .env; set +a
PIDFILE=.dev.pids
case "${1:-up}" in
  up)
    # stale pidfile 자가치유: 프로세스 없으면(run.sh down·크래시·재부팅이 pidfile 잔존) 정리 후 재기동.
    if [ -f "$PIDFILE" ]; then
      alive=""
      while read -r p; do kill -0 "$p" 2>/dev/null && alive=1 || true; done <"$PIDFILE"
      if [ -n "$alive" ]; then echo "already up — ./dev.sh down 먼저"; exit 1; fi
      echo "stale $PIDFILE (프로세스 없음) — 정리하고 재기동"; rm -f "$PIDFILE"
    fi
    mkdir -p .dev-logs
    # mediamtx: 공유 인스턴스(harness-shared-mediamtx) — infra 가 기동(여기선 안 띄움, CEO #115).
    # backend: docker-entrypoint 와 동일하게 alembic upgrade head 후 uvicorn (RULES §9 — alembic 정본).
    #   네이티브 dev 는 공유 mediamtx API 의 호스트 게시포트(127.0.0.1:9997)로 호출(.env 의 docker DNS 값 override).
    #   alembic upgrade head 가 stream_key prefix 데이터 마이그레이션(eb81928ad755)도 함께 적용.
    setsid bash -c "cd backend && source .venv/bin/activate && export MEDIAMTX_API='http://127.0.0.1:9997' && export MEDIAMTX_RTSP_BASE_URL='rtsp://127.0.0.1:8554' && export SYNCED_POSE_STREAM_KEYS='${SYNCED_POSE_STREAM_KEYS:-}' && alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload" >.dev-logs/backend.log 2>&1 & echo $! >>"$PIDFILE"
    # frontend: vite. compose 가 build.args 로 넘기던 VITE_* 를 dev 에선 env 로 주입(viewer=프로젝트 자격증명).
    setsid bash -c "cd frontend && VITE_API_PORT='${BACKEND_PORT}' VITE_MEDIAMTX_WEBRTC_PORT='${MEDIAMTX_WEBRTC_PORT}' VITE_MEDIAMTX_VIEWER_USER='${MEDIAMTX_VIEWER_USER}' VITE_MEDIAMTX_VIEWER_PASS='${MEDIAMTX_VIEWER_PASS:-}' VITE_SYNCED_POSE_STREAM_KEYS='${VITE_SYNCED_POSE_STREAM_KEYS:-${SYNCED_POSE_STREAM_KEYS:-}}' exec npm run dev -- --host 0.0.0.0 --port ${FRONTEND_PORT}" >.dev-logs/frontend.log 2>&1 & echo $! >>"$PIDFILE"
    echo "up — backend :${BACKEND_PORT}, frontend :${FRONTEND_PORT}, 공유 mediamtx webrtc :${MEDIAMTX_WEBRTC_PORT}/api 127.0.0.1:9997 (logs: .dev-logs/)"
    ;;
  down)
    [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
    while read -r pid; do kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; done <"$PIDFILE"
    rm -f "$PIDFILE"
    # 공유 mediamtx 는 건드리지 않는다(다른 프로젝트가 쓰는 외부 인스턴스, CEO #115).
    echo "down"
    ;;
  *) echo "usage: ./dev.sh up|down"; exit 1 ;;
esac
