"""모델별 YOLO worker pool.

같은 모델을 쓰는 카메라는 한 subprocess에서 micro-batch하고, 서로 다른 모델은
모델별 subprocess로 병렬 실행한다. 메인 FastAPI 프로세스는 torch/ultralytics를
import하지 않으며 capture thread는 source별 최신 frame 한 장만 pending에 남긴다.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from app import config
from app.inference.models_dir import is_preset

logger = logging.getLogger("rtsp-keypoint.inference.worker")


@dataclass
class Detection:
    """단일 사람 pose 결과."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    model: str = ""
    # 17 COCO keypoint. 각 점은 request-frame 픽셀 기준 (x, y, confidence).
    keypoints: list[tuple[int, int, float]] = field(default_factory=list)
    # rescue-pose 확장: COCO17 기반 자세 후처리. GPU 추론 경로는 rtsp-keypoint 정본 그대로다.
    posture: Optional[str] = None


@dataclass
class FrameRequest:
    """capture thread에서 worker pool로 보내는 프레임."""

    source_id: str
    frame: np.ndarray
    timestamp: float
    conf_threshold: Optional[float] = None
    model_names: Optional[list[str]] = None
    request_id: int = 0
    imgsz: int = config.INFERENCE_IMGSZ_STAGES[-1]


@dataclass
class InferenceResult:
    """worker pool 결과."""

    source_id: str
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    # WS SEAM: detections 좌표가 기준으로 삼는 실제 request frame 치수.
    frame_w: int = 0
    frame_h: int = 0
    # batch 전체 시간이 아니라 item당 유효 service time. autotuner가 batching 이득을 본다.
    infer_ms: float = 0.0
    idle_ms: float = 0.0
    request_id: int = 0
    model: str = ""
    batch_size: int = 1
    effective_imgsz: int = 0
    oom_recovered: bool = False
    device: str = ""


@dataclass
class _PendingAggregate:
    request: "_AggregateRequest"
    expected_models: set[str]
    created_at: float
    partials: dict[str, InferenceResult] = field(default_factory=dict)


@dataclass(frozen=True)
class _AggregateRequest:
    """결과 조립에 필요한 작은 metadata. 원본 frame을 붙잡아 메모리를 키우지 않는다."""

    source_id: str
    timestamp: float
    request_id: int
    imgsz: int
    model_names: tuple[str, ...]
    frame_w: int
    frame_h: int


def _group_requests_by_imgsz(requests: list[FrameRequest]) -> list[list[FrameRequest]]:
    """같은 imgsz만 한 tensor batch에 넣되 최초 bucket 순서를 보존한다."""
    grouped: dict[int, list[FrameRequest]] = {}
    for request in requests:
        grouped.setdefault(int(request.imgsz), []).append(request)
    return list(grouped.values())


def _merge_partial_results(
    request: FrameRequest | _AggregateRequest,
    partials: list[InferenceResult],
) -> InferenceResult:
    """모델별 partial을 하나의 기존 WS 결과 계약으로 합친다."""
    if isinstance(request, FrameRequest):
        frame_h, frame_w = request.frame.shape[:2]
    else:
        frame_h, frame_w = request.frame_h, request.frame_w
    effective_sizes = [item.effective_imgsz for item in partials if item.effective_imgsz > 0]
    devices = {item.device for item in partials if item.device}
    return InferenceResult(
        source_id=request.source_id,
        timestamp=request.timestamp,
        detections=[detection for item in partials for detection in item.detections],
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        infer_ms=sum(item.infer_ms for item in partials),
        idle_ms=sum(item.idle_ms for item in partials),
        request_id=request.request_id,
        batch_size=max((item.batch_size for item in partials), default=1),
        effective_imgsz=min(effective_sizes, default=int(request.imgsz)),
        oom_recovered=any(item.oom_recovered for item in partials),
        device=next(iter(devices)) if len(devices) == 1 else ",".join(sorted(devices)),
    )


class _ModelLane:
    """한 모델을 한 번만 로드하는 parent-side subprocess lane."""

    def __init__(
        self,
        ctx,
        model_name: str,
        device: Optional[str],
        *,
        batch_max: int,
        batch_timeout_sec: float,
        min_imgsz: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._ctx = ctx
        self._batch_max = max(1, int(batch_max))
        self._batch_timeout_sec = max(0.0, float(batch_timeout_sec))
        self._min_imgsz = max(32, int(min_imgsz))
        self._pending_lock = threading.Lock()
        self._pending: OrderedDict[str, FrameRequest] = OrderedDict()
        self.in_q = None
        self.out_q = None
        self._stop_event = None
        self._proc = None

    def start(self) -> None:
        if self.is_alive():
            return
        self.in_q = self._ctx.Queue(maxsize=max(self._batch_max * 2, self._batch_max + 1))
        self.out_q = self._ctx.Queue(maxsize=max(self._batch_max * 4, 16))
        self._stop_event = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_model_worker_main,
            args=(
                self.model_name,
                self.in_q,
                self.out_q,
                self._stop_event,
                self.device,
                self._batch_max,
                self._batch_timeout_sec,
                self._min_imgsz,
            ),
            daemon=True,
            name=f"yolo-{self.model_name.removesuffix('.pt')}",
        )
        self._proc.start()
        logger.info(
            "Model worker started: model=%s pid=%s device=%s batch=%d/%dms",
            self.model_name,
            self._proc.pid,
            self.device or "auto",
            self._batch_max,
            round(self._batch_timeout_sec * 1000),
        )

    def stop(self) -> None:
        with self._pending_lock:
            self._pending.clear()
        if self._proc is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self.in_q is not None:
            try:
                self.in_q.put_nowait(None)
            except (queue.Full, OSError, ValueError):
                pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            logger.warning("Model worker did not exit gracefully: %s", self.model_name)
            self._proc.terminate()
            self._proc.join(timeout=1)
        self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def enqueue(self, request: FrameRequest) -> Optional[int]:
        """source별 pending 한 장. 기존 source 위치는 유지해 busy source 독점을 막는다."""
        with self._pending_lock:
            displaced = self._pending.get(request.source_id)
            self._pending[request.source_id] = request
        return displaced.request_id if displaced is not None else None

    def flush(self) -> int:
        """round-robin pending을 IPC queue로 이동. queue가 차면 최신값을 parent에 보존한다."""
        if self.in_q is None:
            return 0
        flushed = 0
        while True:
            with self._pending_lock:
                if not self._pending:
                    break
                source_id, request = next(iter(self._pending.items()))
                try:
                    self.in_q.put_nowait(request)
                except queue.Full:
                    break
                except (OSError, ValueError):
                    break
                self._pending.pop(source_id, None)
            flushed += 1
        return flushed

    def drain(self) -> list[InferenceResult]:
        if self.out_q is None:
            return []
        results: list[InferenceResult] = []
        while True:
            try:
                results.append(self.out_q.get_nowait())
            except queue.Empty:
                break
        return results


class InferenceWorker:
    """모델별 process pool facade.

    기존 StreamManager/API가 사용하던 start/stop/submit/drain/config 계약은 유지한다.
    """

    _IN_QUEUE_SIZE = config.INFERENCE_BATCH_MAX * 2
    _OUT_QUEUE_SIZE = max(config.INFERENCE_BATCH_MAX * 4, 16)
    _BATCH_MAX = config.INFERENCE_BATCH_MAX
    _BATCH_TIMEOUT_SEC = config.INFERENCE_BATCH_TIMEOUT_SEC
    _AGGREGATE_TIMEOUT_SEC = config.INFERENCE_AGGREGATE_TIMEOUT_SEC
    _MAX_INFLIGHT_PER_SOURCE = 4

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or config.YOLO_DEFAULT_MODEL
        self.conf_threshold = float(
            conf_threshold if conf_threshold is not None else config.YOLO_CONF_THRESHOLD
        )
        self.device = device or os.getenv("YOLO_DEVICE") or None
        self._ctx = mp.get_context("spawn")
        self._pool_lock = threading.RLock()
        self._lanes: dict[str, _ModelLane] = {}
        self._desired_models: set[str] = {self.model_name}
        self._running = False
        self._enabled = True

        self._aggregate_lock = threading.Lock()
        self._aggregates: dict[int, _PendingAggregate] = {}
        self._aggregate_ids_by_source: dict[str, deque[int]] = {}
        self._ready_results: deque[InferenceResult] = deque()
        self._next_request_id = 1

        self._stats_lock = threading.Lock()
        self._submit_count = 0
        self._drop_count = 0
        # 모델 lane별 실제 service duty EWMA. 합성값은 별도 GPU 조회 없이
        # 1 - product(1 - lane_duty)로 계산해 병렬 lane의 busy union을 근사한다.
        self._duty_lock = threading.Lock()
        self._lane_duty_ewma: dict[str, float] = {}

    def _make_lane(self, model_name: str) -> _ModelLane:
        return _ModelLane(
            self._ctx,
            model_name,
            self.device,
            batch_max=self._BATCH_MAX,
            batch_timeout_sec=self._BATCH_TIMEOUT_SEC,
            min_imgsz=config.INFERENCE_IMGSZ_STAGES[0],
        )

    def _ensure_lane_locked(self, model_name: str) -> _ModelLane:
        lane = self._lanes.get(model_name)
        if lane is None:
            lane = self._make_lane(model_name)
            self._lanes[model_name] = lane
        if self._running and self._enabled:
            lane.start()
        return lane

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        with self._pool_lock:
            self._running = True
            if not self._enabled:
                return
            for model_name in sorted(self._desired_models):
                self._ensure_lane_locked(model_name)

    def stop(self) -> None:
        with self._pool_lock:
            self._running = False
            lanes = list(self._lanes.values())
        for lane in lanes:
            lane.stop()
        with self._aggregate_lock:
            self._aggregates.clear()
            self._aggregate_ids_by_source.clear()
            self._ready_results.clear()
        duty_lock = getattr(self, "_duty_lock", None)
        if duty_lock is not None:
            with duty_lock:
                self._lane_duty_ewma.clear()
        logger.info("Inference worker pool stopped")

    def is_alive(self) -> bool:
        with self._pool_lock:
            if not self._running or not self._enabled or not self._desired_models:
                return True
            return all(
                (lane := self._lanes.get(model_name)) is not None and lane.is_alive()
                for model_name in self._desired_models
            )

    def configure_models(self, model_names: set[str]) -> None:
        """active source가 실제로 요구하는 model lane만 유지해 VRAM을 회수한다."""
        desired = {name for name in model_names if is_preset(name)}
        with self._pool_lock:
            self._desired_models = desired
            if self._running and self._enabled:
                for model_name in sorted(desired):
                    self._ensure_lane_locked(model_name)
            stale = [
                self._lanes.pop(name)
                for name in list(self._lanes)
                if name not in desired
            ]
        for lane in stale:
            lane.stop()
        duty_lock = getattr(self, "_duty_lock", None)
        if duty_lock is not None:
            with duty_lock:
                self._lane_duty_ewma = {
                    name: duty
                    for name, duty in self._lane_duty_ewma.items()
                    if name in desired
                }

    # ── 제출/결과 ───────────────────────────────────────────────

    def _remove_aggregate_locked(self, request_id: int) -> Optional[_PendingAggregate]:
        aggregate = self._aggregates.pop(request_id, None)
        if aggregate is None:
            return None
        source_ids = self._aggregate_ids_by_source.get(aggregate.request.source_id)
        if source_ids is not None:
            try:
                source_ids.remove(request_id)
            except ValueError:
                pass
            if not source_ids:
                self._aggregate_ids_by_source.pop(aggregate.request.source_id, None)
        return aggregate

    def _cancel_aggregate(self, request_id: int) -> bool:
        with self._aggregate_lock:
            return self._remove_aggregate_locked(request_id) is not None

    def submit(self, req: FrameRequest) -> bool:
        """요청을 모델별 lane에 fan-out. 각 lane pending은 source별 latest-wins."""
        with self._pool_lock:
            if not self._enabled:
                return False
            requested = req.model_names if req.model_names else [self.model_name]
            model_names = list(dict.fromkeys(name for name in requested if is_preset(name)))
            if not model_names:
                return False
            request_id = self._next_request_id
            self._next_request_id += 1
            imgsz = min(
                config.INFERENCE_IMGSZ_STAGES,
                key=lambda stage: abs(stage - max(1, int(req.imgsz))),
            )
            aggregate_request = replace(
                req,
                request_id=request_id,
                model_names=model_names,
                imgsz=imgsz,
            )
            frame_h, frame_w = aggregate_request.frame.shape[:2]
            aggregate_meta = _AggregateRequest(
                source_id=aggregate_request.source_id,
                timestamp=aggregate_request.timestamp,
                request_id=request_id,
                imgsz=imgsz,
                model_names=tuple(model_names),
                frame_w=int(frame_w),
                frame_h=int(frame_h),
            )
            bounded_drops = 0
            with self._aggregate_lock:
                source_ids = self._aggregate_ids_by_source.setdefault(
                    req.source_id, deque()
                )
                while len(source_ids) >= self._MAX_INFLIGHT_PER_SOURCE:
                    old_id = source_ids.popleft()
                    if self._aggregates.pop(old_id, None) is not None:
                        bounded_drops += 1
                source_ids.append(request_id)
                self._aggregates[request_id] = _PendingAggregate(
                    request=aggregate_meta,
                    expected_models=set(model_names),
                    created_at=time.monotonic(),
                )

            displaced_ids: set[int] = set()
            for model_name in model_names:
                lane = self._ensure_lane_locked(model_name)
                displaced = lane.enqueue(
                    replace(aggregate_request, model_names=[model_name])
                )
                if displaced:
                    displaced_ids.add(displaced)

        dropped = sum(self._cancel_aggregate(old_id) for old_id in displaced_ids)
        with self._stats_lock:
            self._submit_count += 1
            self._drop_count += dropped + bounded_drops
        return True

    def flush_pending(self) -> int:
        with self._pool_lock:
            lanes = list(self._lanes.values())
        return sum(lane.flush() for lane in lanes)

    def _collect_lane_results(self) -> list[InferenceResult]:
        with self._pool_lock:
            lanes = list(self._lanes.values())
        return [result for lane in lanes for result in lane.drain()]

    def _observe_lane_duty(self, result: InferenceResult) -> bool:
        """유효한 model partial의 service/idle duty를 lane별 bounded EWMA로 기록."""
        model_name = str(getattr(result, "model", ""))
        infer_ms = float(getattr(result, "infer_ms", 0.0))
        idle_ms = float(getattr(result, "idle_ms", 0.0))
        cycle_ms = infer_ms + idle_ms
        if not model_name or infer_ms <= 0.0 or cycle_ms <= 0.0:
            return False
        duty = max(0.0, min(1.0, infer_ms / cycle_ms))
        with self._duty_lock:
            previous = self._lane_duty_ewma.get(model_name)
            if previous is None:
                self._lane_duty_ewma[model_name] = duty
            else:
                alpha = config.AUTOTUNE_EWMA_ALPHA
                self._lane_duty_ewma[model_name] = previous + alpha * (
                    duty - previous
                )
        return True

    def get_pool_duty(self) -> float:
        """활성 모델 lane들의 busy union 근사값(0..1)."""
        duty_lock = getattr(self, "_duty_lock", None)
        if duty_lock is None:
            return 0.0
        with duty_lock:
            duties = tuple(self._lane_duty_ewma.values())
        idle_union = 1.0
        for duty in duties:
            idle_union *= 1.0 - max(0.0, min(1.0, duty))
        return max(0.0, min(1.0, 1.0 - idle_union))

    def drain_results(self) -> list[InferenceResult]:
        self.flush_pending()
        partials = self._collect_lane_results()
        for partial in partials:
            self._observe_lane_duty(partial)
        now = time.monotonic()
        completed: list[InferenceResult] = []
        with self._aggregate_lock:
            for partial in partials:
                aggregate = self._aggregates.get(partial.request_id)
                if aggregate is None or partial.model not in aggregate.expected_models:
                    continue
                aggregate.partials[partial.model] = partial
                if aggregate.expected_models.issubset(aggregate.partials):
                    ordered = [
                        aggregate.partials[name]
                        for name in aggregate.request.model_names or []
                        if name in aggregate.partials
                    ]
                    completed.append(_merge_partial_results(aggregate.request, ordered))
                    self._remove_aggregate_locked(partial.request_id)

            for request_id, aggregate in list(self._aggregates.items()):
                if now - aggregate.created_at < self._AGGREGATE_TIMEOUT_SEC:
                    continue
                if aggregate.partials:
                    ordered = [
                        aggregate.partials[name]
                        for name in aggregate.request.model_names or []
                        if name in aggregate.partials
                    ]
                    completed.append(_merge_partial_results(aggregate.request, ordered))
                self._remove_aggregate_locked(request_id)

            completed.sort(key=lambda result: result.request_id)
            self._ready_results.extend(completed)
            ready = list(self._ready_results)
            self._ready_results.clear()
        return ready

    def get_result(self, timeout: float = 0.0) -> Optional[InferenceResult]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            results = self.drain_results()
            if results:
                if len(results) > 1:
                    with self._aggregate_lock:
                        self._ready_results.extend(results[1:])
                return results[0]
            if timeout <= 0.0 or time.monotonic() >= deadline:
                return None
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    def get_queue_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {"submitted": self._submit_count, "dropped": self._drop_count}

    def get_pool_status(self) -> dict:
        with self._pool_lock:
            return {
                "models": sorted(self._desired_models),
                "workers": {
                    name: {"alive": lane.is_alive()}
                    for name, lane in self._lanes.items()
                },
                "batch_max": self._BATCH_MAX,
                "batch_timeout_ms": self._BATCH_TIMEOUT_SEC * 1000.0,
            }

    # ── 런타임 제어 ──────────────────────────────────────────────

    def set_model(self, model_name: str) -> None:
        if not is_preset(model_name):
            raise ValueError(f"허용되지 않은 모델: {model_name}")
        with self._pool_lock:
            old_model = self.model_name
            self.model_name = model_name
            desired = set(self._desired_models)
        if old_model in desired:
            desired.discard(old_model)
            desired.add(model_name)
            self.configure_models(desired)
        logger.info("Global model switch: %s → %s", old_model, model_name)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._pool_lock:
            if self._enabled == enabled:
                return
            self._enabled = enabled
            lanes = list(self._lanes.values())
            desired = set(self._desired_models)
        if not enabled:
            for lane in lanes:
                lane.stop()
            with self._aggregate_lock:
                self._aggregates.clear()
                self._aggregate_ids_by_source.clear()
                self._ready_results.clear()
            duty_lock = getattr(self, "_duty_lock", None)
            if duty_lock is not None:
                with duty_lock:
                    self._lane_duty_ewma.clear()
        elif self._running:
            self.configure_models(desired)
        logger.info("Inference enabled=%s", enabled)

    def set_conf_threshold(self, threshold: float) -> None:
        self.conf_threshold = float(threshold)

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "model": self.model_name,
            "conf_threshold": self.conf_threshold,
            "device": self.device or "auto",
        }


def _next_lower_imgsz(imgsz: int, minimum: int) -> int:
    stages = [stage for stage in config.INFERENCE_IMGSZ_STAGES if stage >= minimum]
    lower = [stage for stage in stages if stage < imgsz]
    return max(lower, default=minimum)


def _resize_long_side(frame: np.ndarray, imgsz: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    if scale >= 1.0:
        return frame
    import cv2

    return cv2.resize(
        frame,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _rescale_result(
    result: InferenceResult,
    *,
    frame_w: int,
    frame_h: int,
) -> InferenceResult:
    if result.frame_w <= 0 or result.frame_h <= 0:
        return replace(result, frame_w=frame_w, frame_h=frame_h)
    sx = frame_w / result.frame_w
    sy = frame_h / result.frame_h
    detections = [
        replace(
            detection,
            xyxy=(
                round(detection.xyxy[0] * sx),
                round(detection.xyxy[1] * sy),
                round(detection.xyxy[2] * sx),
                round(detection.xyxy[3] * sy),
            ),
            keypoints=[
                (round(x * sx), round(y * sy), confidence)
                for x, y, confidence in detection.keypoints
            ],
        )
        for detection in result.detections
    ]
    return replace(result, detections=detections, frame_w=frame_w, frame_h=frame_h)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def _model_worker_main(
    model_name: str,
    in_q,
    out_q,
    stop_event,
    device_override: Optional[str],
    batch_max: int,
    batch_timeout_sec: float,
    min_imgsz: int,
) -> None:
    """모델 하나를 로드해 source frame을 micro-batch하는 subprocess entry."""
    import torch
    from ultralytics import YOLO

    from app.inference.models_dir import resolve_model_path

    worker_logger = logging.getLogger(f"rtsp-keypoint.inference.worker.{model_name}")
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    path = resolve_model_path(model_name)
    try:
        model = YOLO(path)
        model.to(device)
    except Exception as exc:
        if device.startswith("cuda") and _is_cuda_oom(exc):
            worker_logger.warning("Model load CUDA OOM, CPU fallback: %s", model_name)
            torch.cuda.empty_cache()
            device = "cpu"
            model = YOLO(path)
            model.to(device)
        else:
            worker_logger.exception("Model load failed: %s", model_name)
            return

    measured = False
    idle_started = time.perf_counter()

    def infer_group(
        requests: list[FrameRequest],
        *,
        recovered: bool = False,
    ) -> list[InferenceResult]:
        nonlocal device, measured
        if not requests:
            return []
        conf = min(
            request.conf_threshold
            if request.conf_threshold is not None
            else config.YOLO_CONF_THRESHOLD
            for request in requests
        )
        started = time.perf_counter()
        try:
            predictions = model(
                [request.frame for request in requests],
                conf=conf,
                imgsz=int(requests[0].imgsz),
                device=device,
                verbose=False,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                worker_logger.exception("Inference failed: model=%s", model_name)
                return [
                    InferenceResult(
                        request.source_id,
                        request.timestamp,
                        frame_w=int(request.frame.shape[1]),
                        frame_h=int(request.frame.shape[0]),
                        request_id=request.request_id,
                        model=model_name,
                        effective_imgsz=int(request.imgsz),
                        oom_recovered=recovered,
                        device=device,
                    )
                    for request in requests
                ]

            worker_logger.warning(
                "CUDA OOM recovery: model=%s batch=%d imgsz=%d",
                model_name,
                len(requests),
                requests[0].imgsz,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(requests) > 1:
                middle = len(requests) // 2
                return infer_group(requests[:middle], recovered=True) + infer_group(
                    requests[middle:], recovered=True
                )

            request = requests[0]
            lower = _next_lower_imgsz(int(request.imgsz), min_imgsz)
            if lower < request.imgsz:
                original_h, original_w = request.frame.shape[:2]
                smaller = replace(
                    request,
                    frame=_resize_long_side(request.frame, lower),
                    imgsz=lower,
                )
                retried = infer_group([smaller], recovered=True)
                return [
                    _rescale_result(item, frame_w=original_w, frame_h=original_h)
                    for item in retried
                ]

            if device.startswith("cuda"):
                worker_logger.error(
                    "CUDA OOM at minimum imgsz=%d; model lane CPU fallback: %s",
                    min_imgsz,
                    model_name,
                )
                model.to("cpu")
                device = "cpu"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return infer_group(requests, recovered=True)
            worker_logger.exception("CPU inference OOM: model=%s", model_name)
            return []

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        telemetry_valid = measured
        measured = True
        per_item_ms = elapsed_ms / len(requests) if telemetry_valid else 0.0
        batch_size = len(requests)
        results: list[InferenceResult] = []
        for request, prediction in zip(requests, predictions):
            threshold = (
                request.conf_threshold
                if request.conf_threshold is not None
                else config.YOLO_CONF_THRESHOLD
            )
            # pose 프레임에 사람이 없으면 ultralytics가 keypoints=None을 줄 수 있다.
            # 빈 결과는 정상 경로이며 batch의 다른 프레임 처리에 영향을 주지 않는다.
            if getattr(prediction, "keypoints", None) is None:
                detections = []
            else:
                detections = [
                    detection
                    for detection in _parse_results(prediction, model.names, model_name)
                    if detection.confidence >= threshold
                ]
            results.append(
                InferenceResult(
                    source_id=request.source_id,
                    timestamp=request.timestamp,
                    detections=detections,
                    frame_w=int(request.frame.shape[1]),
                    frame_h=int(request.frame.shape[0]),
                    infer_ms=per_item_ms,
                    request_id=request.request_id,
                    model=model_name,
                    batch_size=batch_size,
                    effective_imgsz=int(request.imgsz),
                    oom_recovered=recovered,
                    device=device,
                )
            )
        return results

    while not stop_event.is_set():
        try:
            first = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if first is None:
            break

        idle_ms = (time.perf_counter() - idle_started) * 1000.0
        latest: OrderedDict[str, FrameRequest] = OrderedDict([(first.source_id, first)])
        deadline = time.perf_counter() + batch_timeout_sec
        while len(latest) < batch_max:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                request = in_q.get(timeout=remaining)
            except queue.Empty:
                break
            if request is None:
                stop_event.set()
                break
            latest[request.source_id] = request

        requests = list(latest.values())
        results: list[InferenceResult] = []
        for group in _group_requests_by_imgsz(requests):
            results.extend(infer_group(group))
        idle_per_item = idle_ms / max(1, len(requests))
        for result in results:
            if result.infer_ms > 0.0:
                result.idle_ms = idle_per_item
            try:
                out_q.put_nowait(result)
            except queue.Full:
                try:
                    out_q.get_nowait()
                    out_q.put_nowait(result)
                except (queue.Empty, queue.Full):
                    pass
        idle_started = time.perf_counter()

    worker_logger.info("Model worker exiting: %s", model_name)


def _parse_results(result, names, model_name: str = "") -> list[Detection]:
    """ultralytics pose Results → box + 17 COCO keypoint Detection."""
    detections: list[Detection] = []
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or getattr(result, "keypoints", None) is None
    ):
        return detections
    boxes = result.boxes
    xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    keypoint_tensor = getattr(result.keypoints, "data", None)
    if keypoint_tensor is None:
        return detections
    keypoint_data = keypoint_tensor.cpu().numpy()
    if (
        keypoint_data.ndim != 3
        or keypoint_data.shape[1] < 17
        or keypoint_data.shape[2] < 3
    ):
        return detections
    for i in range(min(len(boxes), keypoint_data.shape[0])):
        x1, y1, x2, y2 = xyxy_arr[i].tolist()
        cls_id = int(cls_arr[i])
        if hasattr(names, "get"):
            class_name = str(names.get(cls_id, str(cls_id)))
        else:
            class_name = str(names[cls_id]) if cls_id < len(names) else str(cls_id)
        keypoints = [
            (int(x), int(y), float(keypoint_conf))
            for x, y, keypoint_conf in keypoint_data[i, :17, :3]
        ]
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=float(conf_arr[i]),
                xyxy=(x1, y1, x2, y2),
                model=model_name,
                keypoints=keypoints,
                posture=_classify_posture(keypoints),
            )
        )
    return detections


def _angle_three_points(
    a: tuple[int, int, float],
    b: tuple[int, int, float],
    c: tuple[int, int, float],
) -> float:
    """점 b에서 a-b-c가 이루는 각도(degrees)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _classify_posture(kpts: list[tuple[int, int, float]]) -> str:
    """COCO17의 몸통·전체 몸축·관절각을 조합해 세 자세를 분류한다."""
    if len(kpts) < 17:
        return "standing"

    min_confidence = 0.3
    max_lying_torso_angle = 55
    min_lying_hip_angle = 135
    max_flat_body_angle = 55
    max_horizontal_body_angle = 35
    min_horizontal_aspect_ratio = 1.2
    max_sitting_knee_angle = 120
    min_sitting_hip_angle = 55
    max_sitting_hip_angle = 120
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_hip, right_hip = kpts[11], kpts[12]
    left_knee, right_knee = kpts[13], kpts[14]
    left_ankle, right_ankle = kpts[15], kpts[16]

    sides = [
        (left_shoulder, left_hip, left_knee, left_ankle),
        (right_shoulder, right_hip, right_knee, right_ankle),
    ]
    visible_sides = [
        side
        for side in sides
        if side[0][2] > min_confidence and side[1][2] > min_confidence
    ]
    if not visible_sides:
        return "standing"

    def midpoint(points: list[tuple[int, int, float]]) -> tuple[float, float]:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    shoulder_mid = midpoint([side[0] for side in visible_sides])
    hip_mid = midpoint([side[1] for side in visible_sides])

    def axis_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.degrees(math.atan2(abs(b[1] - a[1]), abs(b[0] - a[0])))

    torso_angle = axis_angle(shoulder_mid, hip_mid)

    hip_angles = [
        _angle_three_points(shoulder, hip, knee)
        for shoulder, hip, knee, _ in visible_sides
        if knee[2] > min_confidence
    ]

    visible_ankles = [
        ankle for _, _, _, ankle in visible_sides if ankle[2] > min_confidence
    ]
    visible_knees = [
        knee for _, _, knee, _ in visible_sides if knee[2] > min_confidence
    ]
    lower_points = visible_ankles or visible_knees
    body_angle = (
        axis_angle(shoulder_mid, midpoint(lower_points)) if lower_points else None
    )

    pose_points = [
        point
        for point in kpts[5:7] + kpts[11:17]
        if point[2] > min_confidence
    ]
    pose_width = max(point[0] for point in pose_points) - min(
        point[0] for point in pose_points
    )
    pose_height = max(point[1] for point in pose_points) - min(
        point[1] for point in pose_points
    )
    pose_aspect_ratio = pose_width / max(pose_height, 1)

    # 허리를 숙인 사람은 몸통만 수평이고 어깨→발목 축은 수직에 가깝다. 누운 사람은
    # 골반이 펴져 있거나, 웅크려도 전체 몸축/폭이 바닥 방향으로 놓인다.
    straight_hips = bool(hip_angles) and min(hip_angles) >= min_lying_hip_angle
    flat_body = body_angle is not None and (
        body_angle <= max_horizontal_body_angle
        or (
            body_angle <= max_flat_body_angle
            and pose_aspect_ratio >= min_horizontal_aspect_ratio
        )
    )
    if torso_angle <= max_lying_torso_angle and (straight_hips or flat_body):
        return "lying"

    knee_angles = [
        _angle_three_points(hip, knee, ankle)
        for _, hip, knee, ankle in visible_sides
        if knee[2] > min_confidence and ankle[2] > min_confidence
    ]
    if knee_angles and min(knee_angles) <= max_sitting_knee_angle:
        return "sitting"

    # 발목이 가려진 경우에는 상체가 서 있고 어깨-골반-무릎이 직각에 가까운 다리를
    # 사용한다. 몸통 수평인 허리 숙임에는 이 fallback을 적용하지 않는다.
    partial_hip_angles = [
        _angle_three_points(shoulder, hip, knee)
        for shoulder, hip, knee, ankle in visible_sides
        if knee[2] > min_confidence and ankle[2] <= min_confidence
    ]
    if torso_angle > max_lying_torso_angle and any(
        min_sitting_hip_angle <= angle <= max_sitting_hip_angle
        for angle in partial_hip_angles
    ):
        return "sitting"
    return "standing"
