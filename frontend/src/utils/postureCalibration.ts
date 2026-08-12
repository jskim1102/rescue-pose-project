export interface StandingReferencePayload {
  foot_px: [number, number];
  keypoint_height_px: number;
  height_m: number;
}

export interface PostureCalibrationPayload {
  frame_width: number;
  frame_height: number;
  floor_image_points: Array<[number, number]>;
  floor_world_points: Array<[number, number]>;
  standing_references: StandingReferencePayload[];
}

export interface PostureCalibrationForm {
  frameWidth: string;
  frameHeight: string;
  floorImagePoints: string;
  floorWorldPoints: string;
  standingReferences: string;
}

export type CalibrationPoint = [number, number];

export interface CalibrationPerson {
  keypoints: Array<[number, number, number]>;
}

interface ClientPoint {
  x: number;
  y: number;
}

interface ClientRectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface DetectionFrameSize {
  frameWidth: number;
  frameHeight: number;
}

interface CalibrationFrameSize {
  calibrationWidth: number;
  calibrationHeight: number;
}

export interface FloorAnchorDistances {
  ab: number;
  ac: number;
  bc: number;
  ad: number;
  bd: number;
}

const MIN_REFERENCE_CONFIDENCE = 0.3;
const CORE_KEYPOINT_INDICES = [0, 5, 6, 11, 12, 13, 14, 15, 16];

function parseRows(value: string, columns: number, label: string): number[][] {
  const rows = value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const values = line.split(",").map((part) => Number(part.trim()));
      if (values.length !== columns || values.some((item) => !Number.isFinite(item))) {
        throw new Error(`${label} ${index + 1}행은 숫자 ${columns}개를 쉼표로 구분하세요`);
      }
      return values;
    });
  return rows;
}

export function buildCalibrationPayload(
  form: PostureCalibrationForm,
): PostureCalibrationPayload {
  const frameWidth = Number(form.frameWidth);
  const frameHeight = Number(form.frameHeight);
  if (!Number.isInteger(frameWidth) || frameWidth <= 0 || !Number.isInteger(frameHeight) || frameHeight <= 0) {
    throw new Error("프레임 가로·세로는 양의 정수여야 합니다");
  }

  const imageRows = parseRows(form.floorImagePoints, 2, "화면 바닥점");
  const worldRows = parseRows(form.floorWorldPoints, 2, "실제 바닥점");
  if (imageRows.length < 4 || imageRows.length > 16) {
    throw new Error("바닥 기준점은 4~16개가 필요합니다");
  }
  if (imageRows.length !== worldRows.length) {
    throw new Error("화면 바닥점과 실제 바닥점 개수가 같아야 합니다");
  }

  const referenceRows = parseRows(form.standingReferences, 4, "기립 기준");
  if (referenceRows.length < 3 || referenceRows.length > 20) {
    throw new Error("기립 기준은 서로 다른 위치에서 3~20개가 필요합니다");
  }

  return {
    frame_width: frameWidth,
    frame_height: frameHeight,
    floor_image_points: imageRows.map(([x, y]) => [x, y]),
    floor_world_points: worldRows.map(([x, y]) => [x, y]),
    standing_references: referenceRows.map(([footX, footY, keypointHeight, heightM]) => ({
      foot_px: [footX, footY],
      keypoint_height_px: keypointHeight,
      height_m: heightM,
    })),
  };
}

const formatRows = (rows: number[][]) => rows.map((row) => row.join(",")).join("\n");

export function calibrationToForm(
  calibration: PostureCalibrationPayload,
): PostureCalibrationForm {
  return {
    frameWidth: String(calibration.frame_width),
    frameHeight: String(calibration.frame_height),
    floorImagePoints: formatRows(calibration.floor_image_points),
    floorWorldPoints: formatRows(calibration.floor_world_points),
    standingReferences: formatRows(
      calibration.standing_references.map((reference) => [
        ...reference.foot_px,
        reference.keypoint_height_px,
        reference.height_m,
      ]),
    ),
  };
}

/**
 * object-fit: contain 으로 표시된 영상의 실제 픽셀 영역만 좌표로 바꾼다.
 * 레터박스/필러박스 클릭은 null이라 바닥점으로 저장되지 않는다.
 */
export function mapClientPointToVideo(
  point: ClientPoint,
  rect: ClientRectLike,
  videoWidth: number,
  videoHeight: number,
): CalibrationPoint | null {
  if (
    rect.width <= 0
    || rect.height <= 0
    || videoWidth <= 0
    || videoHeight <= 0
  ) {
    return null;
  }
  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;
  const renderedLeft = rect.left + (rect.width - renderedWidth) / 2;
  const renderedTop = rect.top + (rect.height - renderedHeight) / 2;
  const localX = point.x - renderedLeft;
  const localY = point.y - renderedTop;
  if (localX < 0 || localY < 0 || localX > renderedWidth || localY > renderedHeight) {
    return null;
  }
  return [
    localX / scale,
    localY / scale,
  ];
}

function pointDistance(first: CalibrationPoint, second: CalibrationPoint): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

function trianglePointXy(
  base: number,
  fromA: number,
  fromB: number,
  label: string,
): CalibrationPoint {
  const x = (fromA ** 2 - fromB ** 2 + base ** 2) / (2 * base);
  const heightSquared = fromA ** 2 - x ** 2;
  const tolerance = Math.max(base, fromA, fromB) ** 2 * 1e-9;
  if (heightSquared <= tolerance) {
    throw new Error(`${label} 거리로 유효한 삼각형을 만들 수 없습니다`);
  }
  return [x, Math.sqrt(heightSquared)];
}

function lineSide(first: CalibrationPoint, second: CalibrationPoint, point: CalibrationPoint): number {
  return (
    (second[0] - first[0]) * (point[1] - first[1])
    - (second[1] - first[1]) * (point[0] - first[0])
  );
}

/** 클릭 순서와 무관하게 backend의 convexHull과 같은 바닥 적용 외곽선을 만든다. */
export function convexHullPoints(points: CalibrationPoint[]): CalibrationPoint[] {
  const sorted = [...points]
    .sort((first, second) => first[0] - second[0] || first[1] - second[1])
    .filter((point, index, all) => (
      index === 0 || point[0] !== all[index - 1]?.[0] || point[1] !== all[index - 1]?.[1]
    ));
  if (sorted.length <= 2) return sorted;

  const half = (input: CalibrationPoint[]) => {
    const result: CalibrationPoint[] = [];
    for (const point of input) {
      while (
        result.length >= 2
        && lineSide(result[result.length - 2]!, result[result.length - 1]!, point) <= 0
      ) {
        result.pop();
      }
      result.push(point);
    }
    return result;
  };
  const lower = half(sorted);
  const upper = half([...sorted].reverse());
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/**
 * A=(0,0), B=(AB,0)로 두고 AC·BC 및 AD·BD를 삼각측량한다.
 * D의 Y 부호는 영상에서 C와 D가 AB 선의 같은 편인지 반대편인지로 결정한다.
 */
export function floorWorldPointsFromAnchorDistances(
  imagePoints: CalibrationPoint[],
  distances: FloorAnchorDistances,
): CalibrationPoint[] {
  if (imagePoints.length !== 4) throw new Error("바닥 기준점 A·B·C·D 네 개가 필요합니다");
  const values = Object.values(distances);
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("기준점 사이 실제 거리는 모두 양수여야 합니다");
  }
  const [imageA, imageB, imageC, imageD] = imagePoints;
  if (!imageA || !imageB || !imageC || !imageD || pointDistance(imageA, imageB) <= 1) {
    throw new Error("영상의 A와 B는 서로 떨어진 위치에 찍으세요");
  }
  const sideC = lineSide(imageA, imageB, imageC);
  const sideD = lineSide(imageA, imageB, imageD);
  if (Math.abs(sideC) <= 1 || Math.abs(sideD) <= 1) {
    throw new Error("C와 D는 영상의 A-B 선에서 떨어진 위치에 찍으세요");
  }

  const pointC = trianglePointXy(distances.ab, distances.ac, distances.bc, "A-B-C");
  const pointDPositive = trianglePointXy(distances.ab, distances.ad, distances.bd, "A-B-D");
  const pointD: CalibrationPoint = [
    pointDPositive[0],
    Math.sign(sideC) === Math.sign(sideD) ? pointDPositive[1] : -pointDPositive[1],
  ];
  if (pointDistance(pointC, pointD) <= 1e-6) {
    throw new Error("C와 D가 같은 바닥 좌표로 계산됩니다. 클릭점과 거리를 확인하세요");
  }
  return [[0, 0], [distances.ab, 0], pointC, pointD];
}

export function anchorDistancesFromWorldPoints(
  worldPoints: CalibrationPoint[],
): FloorAnchorDistances {
  if (worldPoints.length < 4) throw new Error("바닥 기준점 네 개가 필요합니다");
  const [a, b, c, d] = worldPoints;
  if (!a || !b || !c || !d) throw new Error("바닥 기준점 네 개가 필요합니다");
  return {
    ab: pointDistance(a, b),
    ac: pointDistance(a, c),
    bc: pointDistance(b, c),
    ad: pointDistance(a, d),
    bd: pointDistance(b, d),
  };
}

function maxPairDistance(points: CalibrationPoint[]): number {
  let maximum = 0;
  for (let firstIndex = 0; firstIndex < points.length; firstIndex += 1) {
    const first = points[firstIndex];
    if (!first) continue;
    for (let secondIndex = firstIndex + 1; secondIndex < points.length; secondIndex += 1) {
      const second = points[secondIndex];
      if (!second) continue;
      maximum = Math.max(maximum, Math.hypot(first[0] - second[0], first[1] - second[1]));
    }
  }
  return maximum;
}

function scaledVisiblePoints(
  person: CalibrationPerson,
  detectionFrame: DetectionFrameSize,
  calibrationFrame: CalibrationFrameSize,
  indices?: number[],
): CalibrationPoint[] {
  if (
    detectionFrame.frameWidth <= 0
    || detectionFrame.frameHeight <= 0
    || calibrationFrame.calibrationWidth <= 0
    || calibrationFrame.calibrationHeight <= 0
  ) {
    return [];
  }
  const scaleX = calibrationFrame.calibrationWidth / detectionFrame.frameWidth;
  const scaleY = calibrationFrame.calibrationHeight / detectionFrame.frameHeight;
  const selected = indices ?? person.keypoints.map((_, index) => index);
  const points: CalibrationPoint[] = [];
  for (const index of selected) {
    const keypoint = person.keypoints[index];
    if (
      !keypoint
      || keypoint[2] <= MIN_REFERENCE_CONFIDENCE
      || !Number.isFinite(keypoint[0])
      || !Number.isFinite(keypoint[1])
    ) {
      continue;
    }
    points.push([keypoint[0] * scaleX, keypoint[1] * scaleY]);
  }
  return points;
}

/**
 * 보정 화면에서 클릭한 사람을 찾고, backend와 같은 core keypoint extent 및 발목 중심을 뽑는다.
 */
export function selectStandingReferenceAtPoint(
  people: CalibrationPerson[],
  clickPoint: CalibrationPoint,
  detectionFrame: DetectionFrameSize,
  calibrationFrame: CalibrationFrameSize,
  heightM: number,
): StandingReferencePayload | null {
  if (!Number.isFinite(heightM) || heightM < 0.5 || heightM > 2.5) return null;

  let selected: CalibrationPerson | null = null;
  let selectedDistance = Number.POSITIVE_INFINITY;
  for (const person of people) {
    const visible = scaledVisiblePoints(person, detectionFrame, calibrationFrame);
    if (visible.length < 5) continue;
    const xs = visible.map((point) => point[0]);
    const ys = visible.map((point) => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = Math.max(20, Math.max(maxX - minX, maxY - minY) * 0.12);
    if (
      clickPoint[0] < minX - padding
      || clickPoint[0] > maxX + padding
      || clickPoint[1] < minY - padding
      || clickPoint[1] > maxY + padding
    ) {
      continue;
    }
    const distance = Math.hypot(
      clickPoint[0] - (minX + maxX) / 2,
      clickPoint[1] - (minY + maxY) / 2,
    );
    if (distance < selectedDistance) {
      selected = person;
      selectedDistance = distance;
    }
  }
  if (!selected) return null;

  const core = scaledVisiblePoints(
    selected,
    detectionFrame,
    calibrationFrame,
    CORE_KEYPOINT_INDICES,
  );
  const ankles = scaledVisiblePoints(selected, detectionFrame, calibrationFrame, [15, 16]);
  if (core.length < 5 || ankles.length === 0) return null;
  const keypointHeight = maxPairDistance(core);
  if (keypointHeight <= 1) return null;
  const foot: CalibrationPoint = [
    ankles.reduce((sum, point) => sum + point[0], 0) / ankles.length,
    ankles.reduce((sum, point) => sum + point[1], 0) / ankles.length,
  ];
  return {
    foot_px: foot,
    keypoint_height_px: keypointHeight,
    height_m: heightM,
  };
}
