/**
 * 클래스별 bbox 기본 색상 — Ultralytics YOLO `Colors` 클래스 와 동일한 20색 팔레트.
 *
 * 같은 class_id 는 항상 같은 색을 받음 (deterministic). 사용자 override 가 있으면 그것을 사용.
 */

export const ULTRALYTICS_PALETTE: readonly string[] = [
  "#FF3838",
  "#FF9D97",
  "#FF701F",
  "#FFB21D",
  "#CFD231",
  "#48F90A",
  "#92CC17",
  "#3DDB86",
  "#1A9334",
  "#00D4BB",
  "#2C99A8",
  "#00C2FF",
  "#344593",
  "#6473FF",
  "#0018EC",
  "#8438FF",
  "#520085",
  "#CB38FF",
  "#FF95C8",
  "#FF37C7",
];

export function defaultClassColor(classId: number): string {
  const i = ((classId % ULTRALYTICS_PALETTE.length) + ULTRALYTICS_PALETTE.length) %
    ULTRALYTICS_PALETTE.length;
  return ULTRALYTICS_PALETTE[i];
}

export function resolveClassColor(
  classId: number,
  override?: Record<number, string>,
): string {
  if (override && classId in override) return override[classId];
  return defaultClassColor(classId);
}

/**
 * COCO 17-keypoint pose 표준 상수 — ultralytics `utils/plotting.py` `Annotator.kpts` 정본 인용
 * (공개 표준, 발명 아님). `KeypointOverlay` 가 스켈레톤 draw 에 소비한다.
 *
 * - POSE_PALETTE: RGB 20색 (ultralytics pose_palette).
 * - SKELETON_EDGES: 19개 엣지, 0-indexed (ultralytics 1-indexed skeleton −1 변환).
 * - KPT_COLOR_IDX / LIMB_COLOR_IDX: 각 점(17)·엣지(19)의 pose_palette 인덱스.
 *   limb 색인 순서 = SKELETON_EDGES 순서 (i번째 엣지 색 = POSE_PALETTE[LIMB_COLOR_IDX[i]]).
 */

export const POSE_PALETTE: readonly [number, number, number][] = [
  [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
  [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
  [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
  [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
];

export const SKELETON_EDGES: readonly [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

export const KPT_COLOR_IDX: readonly number[] = [
  16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9,
];

export const LIMB_COLOR_IDX: readonly number[] = [
  9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16,
];

function rgbString(p: readonly [number, number, number]): string {
  return `rgb(${p[0]}, ${p[1]}, ${p[2]})`;
}

/** 17 keypoint 중 i번째 점 색상 → `rgb(...)`. */
export function kptColor(i: number): string {
  return rgbString(POSE_PALETTE[KPT_COLOR_IDX[i]]);
}

/** 19 skeleton 엣지 중 i번째 엣지 색상 → `rgb(...)`. */
export function limbColor(i: number): string {
  return rgbString(POSE_PALETTE[LIMB_COLOR_IDX[i]]);
}
