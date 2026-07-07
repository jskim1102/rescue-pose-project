"""Ip CAM CRUD characterization + 16대 cap (net-new) 테스트.

CRUD list/create/update/delete/stats 는 추출 코드의 기존 동작을 고정(characterization).
16대 cap 은 net-new 로직 → 명시적 경계 테스트(16 OK, 17번째 409).
"""

import os
import time
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def mtx():
    """ipcam.py 가 호출하는 mediamtx 함수를 mock — 실 httpx 호출 차단.

    create=register_stream(add), update=update_stream(PATCH), delete=remove_stream,
    stats=ensure_stream(self-heal, 현재 path dict|None 반환). ensure 기본값 None(=path 없음) —
    stats 로직을 격리한다 (ensure_stream 자체 동작은 test_mediamtx.py 단위테스트가 검증).
    """
    with patch("app.ipcam.register_stream") as register, \
         patch("app.ipcam.update_stream") as update, \
         patch("app.ipcam.remove_stream") as remove, \
         patch("app.ipcam.ensure_stream") as ensure:
        register.return_value = True
        update.return_value = True
        remove.return_value = True  # 기본 제거 성공 (끝상태 bool — codex #4 orphan-delete)
        ensure.return_value = None
        yield {"register": register, "update": update, "remove": remove, "ensure": ensure}


@pytest.fixture()
def client(mtx):
    """임시 SQLite DB 에 ip_cams 테이블 생성 후 TestClient. 매 테스트 격리."""
    # 임시 DB 파일
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    test_url = f"sqlite:///{path}"

    # 라우터만 단독 마운트 (main.py 전체 배선과 독립 — #37 에서 따로 검증)
    from fastapi import FastAPI

    from app.database import Base, get_db
    from app.ipcam import router as ipcam_router

    app = FastAPI()
    app.include_router(ipcam_router)

    engine = create_engine(test_url, connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    engine.dispose()
    os.unlink(path)


def test_create_returns_201_with_stream_key(client):
    resp = client.post("/api/ipcams", json={"name": "정문", "rtsp_url": "rtsp://x/1"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "정문"
    assert body["rtsp_url"] == "rtsp://x/1"
    # stream_key 는 `ipcam-<hex>` 패턴 — 공유 mediamtx 면 MEDIAMTX_PATH_PREFIX 네임스페이싱이
    # `<prefix>__ipcam-<hex>` 로 앞에 붙는다(운영 기본). prefix 유무에 견고하게 segment 로 검사.
    assert "ipcam-" in body["stream_key"]
    assert "id" in body and "created_at" in body


def test_list_returns_created_cams_in_id_order(client):
    client.post("/api/ipcams", json={"name": "a", "rtsp_url": "rtsp://x/a"})
    client.post("/api/ipcams", json={"name": "b", "rtsp_url": "rtsp://x/b"})
    resp = client.get("/api/ipcams")
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()]
    assert names == ["a", "b"]


def test_list_marks_synced_pose_allowlist(client, monkeypatch):
    import app.ipcam as ipcam_mod

    cam1 = client.post("/api/ipcams", json={"name": "cam1", "rtsp_url": "rtsp://x/1"}).json()
    cam2 = client.post("/api/ipcams", json={"name": "cam2", "rtsp_url": "rtsp://x/2"}).json()
    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_STREAM_KEYS", {cam2["stream_key"]})
    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_DISABLED_STREAM_KEYS", set())

    body = client.get("/api/ipcams").json()

    sync_by_key = {cam["stream_key"]: cam["sync_pose"] for cam in body}
    assert sync_by_key[cam1["stream_key"]] is False
    assert sync_by_key[cam2["stream_key"]] is True


def test_list_auto_marks_direct_poe_sync_and_nvr_overlay(client, monkeypatch):
    """RTSP URL 구조로 direct PoE 계열은 sync_pose, NVR realmonitor 계열은 기존 overlay."""
    import app.ipcam as ipcam_mod

    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_STREAM_KEYS", set())
    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_DISABLED_STREAM_KEYS", set())
    nvr = client.post(
        "/api/ipcams",
        json={
            "name": "nvr",
            "rtsp_url": "rtsp://admin:pw@192.168.0.113:554/cam/realmonitor?channel=2&subtype=0",
        },
    ).json()
    poe = client.post(
        "/api/ipcams",
        json={"name": "poe", "rtsp_url": "rtsp://192.168.0.152:8554/camera01"},
    ).json()

    body = client.get("/api/ipcams").json()

    sync_by_key = {cam["stream_key"]: cam["sync_pose"] for cam in body}
    assert sync_by_key[nvr["stream_key"]] is False
    assert sync_by_key[poe["stream_key"]] is True


def test_list_sync_pose_disable_override_beats_auto(client, monkeypatch):
    import app.ipcam as ipcam_mod

    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_STREAM_KEYS", set())
    cam = client.post(
        "/api/ipcams",
        json={"name": "poe", "rtsp_url": "rtsp://192.168.0.152:8554/camera01"},
    ).json()
    monkeypatch.setattr(ipcam_mod, "SYNCED_POSE_DISABLED_STREAM_KEYS", {cam["stream_key"]})

    body = client.get("/api/ipcams").json()

    assert {c["stream_key"]: c["sync_pose"] for c in body}[cam["stream_key"]] is False


def test_update_changes_name_and_url(client):
    cam = client.post("/api/ipcams", json={"name": "old", "rtsp_url": "rtsp://x/old"}).json()
    resp = client.put(f"/api/ipcams/{cam['id']}", json={"name": "new", "rtsp_url": "rtsp://x/new"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "new"
    assert resp.json()["rtsp_url"] == "rtsp://x/new"


def test_update_missing_returns_404(client):
    resp = client.put("/api/ipcams/999", json={"name": "x", "rtsp_url": "rtsp://x/x"})
    assert resp.status_code == 404


def test_delete_removes_cam(client):
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 204
    assert client.get("/api/ipcams").json() == []


def test_delete_missing_returns_404(client):
    resp = client.delete("/api/ipcams/999")
    assert resp.status_code == 404


# ─── net-new: mediamtx 사이드이펙트 배선 (test-first) ───


def test_create_registers_mediamtx_path(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://x/c"}).json()
    mtx["register"].assert_called_once_with(cam["stream_key"], "rtsp://x/c")


def test_detection_ws_captures_from_mediamtx_fanout(client, monkeypatch):
    """좌표 WS 는 원본 RTSP 재접속 대신 mediamtx path RTSP 를 읽어 WHEP 영상과 시간축을 맞춘다."""
    from app import ipcam as ipcam_mod

    stream_key = "rescue_pose__ipcam-test"
    cam = SimpleNamespace(stream_key=stream_key, rtsp_url="rtsp://camera/raw")
    replaces = []
    starts = []
    stops = []

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return cam

    class FakeDb:
        def query(self, model):
            return FakeQuery()

        def close(self):
            return None

    class FakeStreamManager:
        def set_source_annotated_frame(self, sid, enabled):
            return None

        def replace_source(self, sid, source):
            replaces.append((sid, source))
            return True

        def start_capture(self, sid, source):
            starts.append((sid, source))
            return True

        def get_source_latest_detections(self, sid):
            return None

        def stop_capture(self, sid):
            stops.append(sid)

    monkeypatch.setattr(ipcam_mod, "stream_manager", FakeStreamManager())
    monkeypatch.setattr(ipcam_mod, "MEDIAMTX_DETECTION_FANOUT_STREAM_KEYS", {stream_key})
    monkeypatch.setattr("app.database.SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(
        ipcam_mod,
        "stream_rtsp_url",
        lambda stream_key: f"rtsp://backend:s3cret@mtx.local:8554/{stream_key}",
    )

    with client.websocket_connect(f"/api/ipcams/{stream_key}/ws"):
        deadline = time.monotonic() + 1.0
        while not starts and time.monotonic() < deadline:
            time.sleep(0.01)

    assert replaces == [
        (
            f"ipcam-{stream_key}",
            f"rtsp://backend:s3cret@mtx.local:8554/{stream_key}",
        )
    ]
    assert starts == [
        (
            f"ipcam-{stream_key}",
            f"rtsp://backend:s3cret@mtx.local:8554/{stream_key}",
        )
    ]
    assert stops == [f"ipcam-{stream_key}"]


def test_detection_ws_defaults_to_camera_rtsp(client, monkeypatch):
    """fan-out allowlist 에 없는 카메라는 원본 RTSP direct 캡처를 유지한다(NVR 지연 회귀 방지)."""
    from app import ipcam as ipcam_mod

    stream_key = "rescue_pose__ipcam-nvr"
    cam = SimpleNamespace(stream_key=stream_key, rtsp_url="rtsp://nvr/cam1")
    starts = []

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return cam

    class FakeDb:
        def query(self, model):
            return FakeQuery()

        def close(self):
            return None

    class FakeStreamManager:
        def set_source_annotated_frame(self, sid, enabled):
            return None

        def replace_source(self, sid, source):
            starts.append(("replace", sid, source))
            return True

        def start_capture(self, sid, source):
            starts.append(("start", sid, source))
            return True

        def get_source_latest_detections(self, sid):
            return None

        def stop_capture(self, sid):
            return None

    monkeypatch.setattr(ipcam_mod, "stream_manager", FakeStreamManager())
    monkeypatch.setattr(ipcam_mod, "MEDIAMTX_DETECTION_FANOUT_STREAM_KEYS", set())
    monkeypatch.setattr("app.database.SessionLocal", lambda: FakeDb())

    with client.websocket_connect(f"/api/ipcams/{stream_key}/ws"):
        deadline = time.monotonic() + 1.0
        while not starts and time.monotonic() < deadline:
            time.sleep(0.01)

    assert starts[:2] == [
        ("replace", f"ipcam-{stream_key}", "rtsp://nvr/cam1"),
        ("start", f"ipcam-{stream_key}", "rtsp://nvr/cam1"),
    ]


def test_update_rtsp_change_updates_via_patch(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "u", "rtsp_url": "rtsp://x/old"}).json()
    mtx["update"].reset_mock()
    mtx["remove"].reset_mock()
    client.put(f"/api/ipcams/{cam['id']}", json={"name": "u", "rtsp_url": "rtsp://x/new"})
    # 원자 PATCH 갱신 — teardown(remove) 선행 없음 (카메라 암전 0)
    mtx["update"].assert_called_once_with(cam["stream_key"], "rtsp://x/new")
    mtx["remove"].assert_not_called()


def test_update_no_rtsp_change_no_reregister(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "u", "rtsp_url": "rtsp://x/same"}).json()
    mtx["update"].reset_mock()
    mtx["remove"].reset_mock()
    # name 만 변경, rtsp 동일 → mediamtx 갱신 안 함
    client.put(f"/api/ipcams/{cam['id']}", json={"name": "renamed", "rtsp_url": "rtsp://x/same"})
    mtx["update"].assert_not_called()
    mtx["remove"].assert_not_called()


def test_delete_removes_mediamtx_path(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].reset_mock()
    client.delete(f"/api/ipcams/{cam['id']}")
    # 초기 제거 + commit 후 sweep = 2회, 둘 다 같은 stream_key (TOCTOU race close).
    assert mtx["remove"].call_count == 2
    for call in mtx["remove"].call_args_list:
        assert call.args == (cam["stream_key"],)


# ─── codex #4 — mediamtx 제거 실패 시 orphan 라이브 스트림 차단 (security) ───


def test_delete_aborts_when_mediamtx_remove_fails(client, mtx):
    """remove_stream=False → 503 + DB row 유지(commit 안 됨).

    DB row 를 지우면 mediamtx path 가 살아남아 삭제된 카메라가 URL 로 계속 재생되는
    orphan 라이브 스트림이 된다 — 제거 끝상태 확인 전엔 삭제하지 않는다.
    """
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].return_value = False
    resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 503
    assert len(client.get("/api/ipcams").json()) == 1


def test_delete_503_when_mediamtx_api_unset(client, mtx):
    """MEDIAMTX_API 미설정(remove_stream 이 RuntimeError) → 503 + DB row 유지."""
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].side_effect = RuntimeError("MEDIAMTX_API 미설정")
    resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 503
    assert len(client.get("/api/ipcams").json()) == 1


def test_delete_sweeps_path_after_commit_to_close_resurrection_race(client, mtx):
    """commit 직후 sweep remove_stream 1회 더 — 삭제~commit 사이 stats 폴링 ensure_stream 이
    path 를 재등록(resurrection)했어도 sweep 가 제거하고, row 가 없어 다시 부활 안 한다(TOCTOU)."""
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].reset_mock()
    resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 204
    assert mtx["remove"].call_count == 2  # 초기 + sweep
    assert client.get("/api/ipcams").json() == []


def test_delete_succeeds_when_sweep_fails(client, mtx):
    """sweep(2번째 remove)이 실패해도 — commit 은 이미 끝났으므로 — 삭제는 204 로 성공한다
    (best-effort; orphan 가능성은 warning 로그만). 초기 제거는 성공해야 commit 에 도달."""
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].side_effect = [True, False]
    resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 204
    assert mtx["remove"].call_count == 2
    assert client.get("/api/ipcams").json() == []


# ─── net-new: stats mediamtx /v3/paths/get 기반 (test-first) ───


def test_stats_active_with_readers(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "s", "rtsp_url": "rtsp://x/s"}).json()
    # ensure_stream(self-heal) 이 조회한 path 를 그대로 반환 — ready=true + readers 2개
    mtx["ensure"].return_value = {"ready": True, "readers": [{}, {}]}
    resp = client.get(f"/api/ipcams/{cam['stream_key']}/stats")
    assert resp.status_code == 200
    assert resp.json() == {"active": True, "readers": 2}


def test_stats_inactive_when_path_missing(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "s", "rtsp_url": "rtsp://x/s"}).json()
    # path 없음 → None
    mtx["ensure"].return_value = None
    resp = client.get(f"/api/ipcams/{cam['stream_key']}/stats")
    assert resp.status_code == 200
    assert resp.json() == {"active": False, "readers": 0}


def test_stats_inactive_when_not_ready(client, mtx):
    cam = client.post("/api/ipcams", json={"name": "s", "rtsp_url": "rtsp://x/s"}).json()
    # path 존재하나 아직 ready 아님 (source 연결 전)
    mtx["ensure"].return_value = {"ready": False, "readers": []}
    resp = client.get(f"/api/ipcams/{cam['stream_key']}/stats")
    assert resp.status_code == 200
    assert resp.json() == {"active": False, "readers": 0}


def test_stats_invokes_ensure_stream_selfheal(client, mtx):
    """stats 폴링이 ensure_stream(self-heal)을 호출 — mediamtx recreate 후 path 자가복구 접점.
    프론트가 이 엔드포인트를 폴링하므로 backend 재시작 없이 path 가 복원된다."""
    cam = client.post("/api/ipcams", json={"name": "s", "rtsp_url": "rtsp://x/s"}).json()
    mtx["ensure"].reset_mock()
    mtx["ensure"].return_value = {"ready": True, "readers": []}
    resp = client.get(f"/api/ipcams/{cam['stream_key']}/stats")
    assert resp.status_code == 200
    mtx["ensure"].assert_called_once()
    assert mtx["ensure"].call_args.args[1] == cam["stream_key"]


# ─── net-new: 16대 cap (test-first) ───


def test_can_register_up_to_max_ipcams(client):
    from app.config import MAX_IPCAMS

    for i in range(MAX_IPCAMS):
        resp = client.post("/api/ipcams", json={"name": f"cam{i}", "rtsp_url": f"rtsp://x/{i}"})
        assert resp.status_code == 201, f"등록 {i+1}/{MAX_IPCAMS} 실패: {resp.text}"
    assert len(client.get("/api/ipcams").json()) == MAX_IPCAMS


def test_register_over_cap_returns_409(client):
    from app.config import MAX_IPCAMS

    for i in range(MAX_IPCAMS):
        client.post("/api/ipcams", json={"name": f"cam{i}", "rtsp_url": f"rtsp://x/{i}"})

    # MAX_IPCAMS+1 번째 → 409
    resp = client.post("/api/ipcams", json={"name": "over", "rtsp_url": "rtsp://x/over"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == f"최대 {MAX_IPCAMS}대까지 등록할 수 있습니다"


# ─── net-new: rtsp:// 스킴 검증 (injection 2중 방어 1단; 셸안전은 shlex.quote) ───


def test_create_rejects_non_rtsp_scheme_400(client):
    resp = client.post("/api/ipcams", json={"name": "bad", "rtsp_url": "http://x/stream"})
    assert resp.status_code == 400
    assert client.get("/api/ipcams").json() == []  # DB 오염 없음


def test_update_rejects_non_rtsp_scheme_400(client):
    cam = client.post("/api/ipcams", json={"name": "ok", "rtsp_url": "rtsp://x/ok"}).json()
    resp = client.put(f"/api/ipcams/{cam['id']}", json={"name": "ok", "rtsp_url": "ftp://x/evil"})
    assert resp.status_code == 400
    assert client.get("/api/ipcams").json()[0]["rtsp_url"] == "rtsp://x/ok"  # 기존 유지


def test_create_accepts_rtsp_url_with_metachars(client):
    # rtsp:// 스킴이면 셸 메타문자가 있어도 등록 성공 — shlex.quote 가 중립화.
    resp = client.post("/api/ipcams", json={"name": "ok", "rtsp_url": "rtsp://u:p@10.0.0.1:554/stream"})
    assert resp.status_code == 201


# ─── net-new: P1-3 mediamtx 등록 실패 시 rollback (성공처럼 보이는 실패 방지) ───


def test_create_rolls_back_when_mediamtx_register_fails(client, mtx):
    """register_stream=False → 503 + DB 미저장(롤백)."""
    mtx["register"].return_value = False
    resp = client.post("/api/ipcams", json={"name": "x", "rtsp_url": "rtsp://x/1"})
    assert resp.status_code == 503
    assert client.get("/api/ipcams").json() == []  # 롤백되어 흔적 없음


def test_create_503_when_mediamtx_api_unset(client, mtx):
    """MEDIAMTX_API 미설정(register_stream 이 RuntimeError) → 503 + 롤백."""
    mtx["register"].side_effect = RuntimeError("MEDIAMTX_API 미설정")
    resp = client.post("/api/ipcams", json={"name": "x", "rtsp_url": "rtsp://x/1"})
    assert resp.status_code == 503
    assert client.get("/api/ipcams").json() == []


# ─── net-new: P1-4 자격증명 마스킹 ───


def test_list_masks_rtsp_credentials(client):
    """목록 응답의 rtsp_url 비밀번호는 *** 로 마스킹 (평문 노출 차단)."""
    client.post("/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://admin:secret@10.0.0.5:554/s"})
    url = client.get("/api/ipcams").json()[0]["rtsp_url"]
    assert url == "rtsp://admin:***@10.0.0.5:554/s"
    assert "secret" not in url


def test_update_masked_url_preserves_credentials(client, mtx):
    """수정 시 마스킹된 URL 그대로 들어오면(미수정) 기존 자격증명 유지 + 재등록 안 함."""
    cam = client.post(
        "/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://admin:secret@10.0.0.5:554/s"}
    ).json()
    masked = cam["rtsp_url"]  # rtsp://admin:***@10.0.0.5:554/s
    mtx["register"].reset_mock()
    mtx["remove"].reset_mock()

    # 이름만 변경, URL 은 마스킹된 값 그대로 → 자격증명 보존, mediamtx 재등록 없음
    resp = client.put(f"/api/ipcams/{cam['id']}", json={"name": "renamed", "rtsp_url": masked})
    assert resp.status_code == 200
    assert resp.json()["name"] == "renamed"
    assert resp.json()["rtsp_url"] == masked  # 여전히 마스킹 (저장된 실값은 secret 그대로)
    mtx["update"].assert_not_called()
    mtx["remove"].assert_not_called()
    mtx["register"].assert_not_called()


def test_update_masked_password_with_changed_host_rejected_400(client, mtx):
    """credential-exfil 가드(CEO #86, security HIGH): 비번을 *** 그대로 두고 host(또는
    비번 외 컴포넌트)만 바꿔 저장하는 것을 **400 으로 거부**한다.

    이전 동작(CEO #56-1, "주소만 바꿔도 반영")은 old 실비번을 **새 host 로** 복원·재등록했는데,
    그건 사용자가 모르는 호스트로 실비번이 재전송되는 credential-exfil 표면이다(예: `:***@공격자host`,
    API 인증 없음). 가드는 비번 외 컴포넌트가 바뀌었으면 평문 비번 재입력을 요구한다 — 동일 주소일
    때만 *** 복원 허용. (#56-1 의 "주소만 바꿔 반영"은 이 보안 가드로 의도적으로 철회됨.)
    """
    cam = client.post(
        "/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://admin:secret@10.0.0.5:554/s"}
    ).json()
    masked = cam["rtsp_url"]
    assert masked == "rtsp://admin:***@10.0.0.5:554/s"
    mtx["register"].reset_mock()
    mtx["remove"].reset_mock()

    # host 만 .5 → .9 로 변경, 비번은 *** 그대로 → 거부(평문 재입력 요구)
    changed = "rtsp://admin:***@10.0.0.9:554/s"
    resp = client.put(f"/api/ipcams/{cam['id']}", json={"name": "c", "rtsp_url": changed})
    assert resp.status_code == 400
    # mediamtx 무변경(재등록·제거 0), DB 도 기존 .5 유지(주소 안 바뀜 = 실비번 안 샘)
    mtx["register"].assert_not_called()
    mtx["update"].assert_not_called()
    mtx["remove"].assert_not_called()
    assert client.get("/api/ipcams").json()[0]["rtsp_url"] == "rtsp://admin:***@10.0.0.5:554/s"


# ─── net-new: P1 보안 핫픽스 — 예약문자(/ #) 비번 마스킹 누수 (CEO #60) ───


def test_list_masks_password_with_reserved_chars_no_leak(client):
    """버그재현(CEO #60): 비번에 / 나 # 가 있으면 urlsplit 이 비번을 path/fragment 로
    끊어, 마스킹돼도 비번 일부가 GET 응답에 평문 누수했다(인증 없는 API → 비번 노출).
    last-@ 마스킹은 예약문자가 있어도 비번 전체를 *** 로 가린다.
    """
    pw = "pa/ss#word"
    client.post(
        "/api/ipcams", json={"name": "c", "rtsp_url": f"rtsp://admin:{pw}@10.0.0.5:554/cam"}
    )
    url = client.get("/api/ipcams").json()[0]["rtsp_url"]
    assert url == "rtsp://admin:***@10.0.0.5:554/cam"
    # 비번의 어떤 조각도 응답에 남으면 안 됨 (path/fragment 누수 포함)
    for frag in (pw, "ss", "word"):
        assert frag not in url, f"비번 조각 누수: {frag!r} in {url!r}"


def test_update_masked_reserved_char_password_changed_host_rejected_400(client, mtx):
    """/·# 비번 카메라도 *** 유지한 채 host 만 바꾸면 동일하게 400 거부 — 예약문자 비번이라도
    credential-exfil 가드가 평문 재입력을 요구한다(CEO #86)."""
    cam = client.post(
        "/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://admin:pa/ss#x@10.0.0.5:554/cam"}
    ).json()
    assert cam["rtsp_url"] == "rtsp://admin:***@10.0.0.5:554/cam"
    mtx["register"].reset_mock()
    mtx["remove"].reset_mock()

    resp = client.put(
        f"/api/ipcams/{cam['id']}",
        json={"name": "c", "rtsp_url": "rtsp://admin:***@10.0.0.7:554/cam"},
    )
    assert resp.status_code == 400
    mtx["register"].assert_not_called()
    mtx["update"].assert_not_called()
    mtx["remove"].assert_not_called()
    # DB 도 기존 .5 유지 (실비번 .7 로 안 샘)
    assert client.get("/api/ipcams").json()[0]["rtsp_url"] == "rtsp://admin:***@10.0.0.5:554/cam"


def test_update_new_plaintext_password_with_host_change_allowed(client, mtx):
    """새 평문 비번을 입력하면(= *** 아님) host 변경도 허용 — 가드는 *** 일 때만 작동.
    사용자가 실비번을 다시 제출했으므로 exfil 위험 없음(CEO #86 양성 케이스)."""
    cam = client.post(
        "/api/ipcams", json={"name": "c", "rtsp_url": "rtsp://admin:secret@10.0.0.5:554/s"}
    ).json()
    mtx["register"].reset_mock()
    mtx["remove"].reset_mock()
    # 새 비번 newpw + host .9 → 200, 새 host+새 비번 전체로 PATCH 갱신(remove 선행 없음)
    resp = client.put(
        f"/api/ipcams/{cam['id']}",
        json={"name": "c", "rtsp_url": "rtsp://admin:newpw@10.0.0.9:554/s"},
    )
    assert resp.status_code == 200
    mtx["update"].assert_called_once_with(cam["stream_key"], "rtsp://admin:newpw@10.0.0.9:554/s")
    mtx["remove"].assert_not_called()
    assert client.get("/api/ipcams").json()[0]["rtsp_url"] == "rtsp://admin:***@10.0.0.9:554/s"


def test_mask_restore_roundtrip_special_char_passwords():
    """mask→restore 가 특수·예약문자 비번을 누수 없이 왕복하는지 (헬퍼 직접 단위).

    수용기준 문자셋: @ : % 공백 ! / # (공백은 API _validate 가 막지만 헬퍼는 견고해야 함).
    """
    from app.ipcam import _restore_masked_password, mask_rtsp_credentials

    for pw in ["p@ss", "pa:ss", "p%2Fss", "pa ss", "p!ss", "pa/ss", "pa#ss", "a/b#c?d@e"]:
        original = f"rtsp://admin:{pw}@10.0.0.5:554/cam"
        masked = mask_rtsp_credentials(original)
        assert masked == "rtsp://admin:***@10.0.0.5:554/cam", f"{pw!r} 마스킹 불완전: {masked!r}"
        assert pw not in masked, f"{pw!r} 평문 누수: {masked!r}"
        # 마스킹된 값으로(주소 미변경) 저장 → 실제 비번 복원 = 원본
        restored = _restore_masked_password(masked, original)
        assert restored == original, f"{pw!r} 복원 실패: {restored!r}"


# ─── net-new: P1 — validate 에러 메시지(→400 응답 본문)에 비번 평문 반사 (CEO #66) ───


def test_create_forbidden_char_400_masks_password_in_detail(client):
    """버그재현(CEO #66): 금지문자+비번 url 등록 → 400 응답 본문에 비번 평문 반사됐다.
    이제 에러 메시지의 url 도 마스킹된다(***). (`;` = 셸메타 금지문자, 비번에 위치)
    """
    resp = client.post(
        "/api/ipcams", json={"name": "x", "rtsp_url": "rtsp://admin:se;cret@10.0.0.5:554/cam"}
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "se;cret" not in detail and "secret" not in detail  # 평문 비번 0
    assert "***" in detail  # 마스킹된 형태로 들어감
    assert client.get("/api/ipcams").json() == []  # DB 오염 없음


def test_create_bad_scheme_400_masks_password_in_detail(client):
    """스킴 에러 메시지도 비번 마스킹 (http:// 인데 자격증명 포함된 경우)."""
    resp = client.post(
        "/api/ipcams", json={"name": "x", "rtsp_url": "http://admin:secret@10.0.0.5/cam"}
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "secret" not in detail
    assert "***" in detail


def test_update_forbidden_char_400_masks_password_in_detail(client):
    """update 경로의 validate 에러도 동일하게 비번 마스킹."""
    cam = client.post(
        "/api/ipcams", json={"name": "ok", "rtsp_url": "rtsp://admin:good@10.0.0.5:554/s"}
    ).json()
    resp = client.put(
        f"/api/ipcams/{cam['id']}",
        json={"name": "ok", "rtsp_url": "rtsp://admin:ba|d@10.0.0.9:554/s"},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "ba|d" not in detail and "***" in detail


def test_concurrent_create_at_cap_never_exceeds_max():
    """두 요청이 동시에 count<MAX 를 관찰하고 둘 다 insert 해 cap 을 초과하는 race 를 재현한다.

    파일기반 SQLite(:memory: 는 커넥션/스레드간 공유 불가) + MAX_IPCAMS=1 로 패치, 두 스레드가
    threading.Barrier 로 동시에 create_ipcam 을 호출한다(각자 별도 Session/커넥션). register_stream
    은 짧게 sleep 후 성공하도록 mock — 선행 스레드가 commit 하기 전에 후행 스레드가 count 를
    관찰(수정 전)하거나 BEGIN IMMEDIATE 에서 대기(수정 후)하도록 race 창을 넓혀 결정적으로 만든다.

    수정 전(락 없음): 두 스레드가 모두 count=0 을 보고 둘 다 insert → 2행(cap 초과), 409 없음.
    수정 후(BEGIN IMMEDIATE): 후행 스레드가 락 대기 후 갱신된 count=1 을 관찰 → 409 → 1행.
    """
    import threading
    import time

    from fastapi import HTTPException

    import app.ipcam as ipcam
    from app.database import Base
    from app.ipcam import IpCamCreate, create_ipcam
    from app.models import IpCam

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 5}
    )
    TestSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    barrier = threading.Barrier(2)
    results: dict[int, tuple] = {}

    def slow_register(stream_key, rtsp_url):
        time.sleep(0.3)  # commit 전 창을 넓혀 race 를 결정적으로 재현
        return True

    def worker(idx: int) -> None:
        db = TestSession()
        try:
            barrier.wait()
            resp = create_ipcam(IpCamCreate(name=f"c{idx}", rtsp_url=f"rtsp://x/{idx}"), db)
            results[idx] = ("ok", resp.id)
        except HTTPException as e:
            results[idx] = ("http", e.status_code)
        finally:
            db.close()

    try:
        with patch.object(ipcam, "MAX_IPCAMS", 1), \
             patch.object(ipcam, "register_stream", side_effect=slow_register):
            threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        final = TestSession().query(IpCam).count()
        # 핵심 불변식: 어떤 동시성 인터리빙에서도 cap 을 넘겨 insert 되지 않는다.
        assert final == 1, f"cap 초과: {final}행 (동시 insert race)"
        # 패자는 409 를 받는다 (F7 계약이 직렬 요청뿐 아니라 동시 요청에서도 유지).
        statuses = sorted(v[1] for v in results.values() if v[0] == "http")
        assert statuses == [409], f"패자가 409 를 받지 못함: {results}"
    finally:
        engine.dispose()
        os.unlink(path)


# ─── CEO #280: 추론 캡처 lifecycle 배선 (F1 replace / F2 force-remove) ───


def test_update_rtsp_change_replaces_inference_capture(client, mtx):
    """F1(fix15): URL 변경 시 update_ipcam 이 stream_manager.replace_source 로 옛 캡처를
    새 source 로 교체(옛 RTSP 잔존 X). fan-out 미지정 카메라는 원본 RTSP direct 를 유지."""
    cam = client.post("/api/ipcams", json={"name": "u", "rtsp_url": "rtsp://x/old"}).json()
    with patch("app.ipcam.stream_manager") as sm:
        resp = client.put(
            f"/api/ipcams/{cam['id']}", json={"name": "u", "rtsp_url": "rtsp://x/new"}
        )
    assert resp.status_code == 200
    sm.replace_source.assert_called_once_with(f"ipcam-{cam['stream_key']}", "rtsp://x/new")


def test_update_rtsp_change_replaces_fanout_capture_when_allowlisted(client, mtx, monkeypatch):
    """fan-out allowlist 카메라만 URL 변경 후 mediamtx fan-out 캡처를 유지한다."""
    import app.ipcam as ipcam_mod

    cam = client.post("/api/ipcams", json={"name": "u", "rtsp_url": "rtsp://x/old"}).json()
    monkeypatch.setattr(ipcam_mod, "MEDIAMTX_DETECTION_FANOUT_STREAM_KEYS", {cam["stream_key"]})
    with patch("app.ipcam.stream_manager") as sm, \
         patch("app.ipcam.stream_rtsp_url", return_value="rtsp://backend:s3cret@mtx.local:8554/path"):
        resp = client.put(
            f"/api/ipcams/{cam['id']}", json={"name": "u", "rtsp_url": "rtsp://x/new"}
        )
    assert resp.status_code == 200
    sm.replace_source.assert_called_once_with(
        f"ipcam-{cam['stream_key']}",
        "rtsp://backend:s3cret@mtx.local:8554/path",
    )


def test_update_no_rtsp_change_does_not_replace_capture(client, mtx):
    """rtsp 동일(이름만 변경)이면 replace_source 안 함 — 불필요 캡처 재생성 없음."""
    cam = client.post("/api/ipcams", json={"name": "u", "rtsp_url": "rtsp://x/same"}).json()
    with patch("app.ipcam.stream_manager") as sm:
        client.put(
            f"/api/ipcams/{cam['id']}", json={"name": "renamed", "rtsp_url": "rtsp://x/same"}
        )
    sm.replace_source.assert_not_called()


def test_delete_force_removes_inference_capture(client, mtx):
    """F2(fix16): 삭제 시 delete_ipcam 이 stream_manager.remove_capture(ref 무시 force-stop)로
    캡처 완전 제거 — viewer 2+여도 스레드 잔존 X."""
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    with patch("app.ipcam.stream_manager") as sm:
        resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 204
    # 삭제 확정 후(mediamtx 성공 뒤) + commit 후 sweep = 2회, 둘 다 같은 source_id (대칭 정리).
    assert sm.remove_capture.call_count == 2
    for call in sm.remove_capture.call_args_list:
        assert call.args == (f"ipcam-{cam['stream_key']}",)


def test_delete_503_does_not_remove_capture(client, mtx):
    """mediamtx 제거 실패(503)면 캡처를 건드리지 않는다 — 삭제 안 된 카메라의 검출 보존
    (remove_capture 는 mediamtx 성공 후에만 호출)."""
    cam = client.post("/api/ipcams", json={"name": "d", "rtsp_url": "rtsp://x/d"}).json()
    mtx["remove"].return_value = False
    with patch("app.ipcam.stream_manager") as sm:
        resp = client.delete(f"/api/ipcams/{cam['id']}")
    assert resp.status_code == 503
    sm.remove_capture.assert_not_called()
