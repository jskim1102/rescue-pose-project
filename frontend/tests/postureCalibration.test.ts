import assert from "node:assert/strict";
import test from "node:test";

import {
  anchorDistancesFromWorldPoints,
  buildCalibrationPayload,
  calibrationToForm,
  convexHullPoints,
  floorWorldPointsFromAnchorDistances,
  mapClientPointToVideo,
  selectStandingReferenceAtPoint,
} from "../src/utils/postureCalibration.ts";


test("보정 폼의 줄 단위 좌표를 API payload로 변환한다", () => {
  const payload = buildCalibrationPayload({
    frameWidth: "1920",
    frameHeight: "1080",
    floorImagePoints: "100,900\n1800,900\n1600,500\n300,500",
    floorWorldPoints: "0,0\n6,0\n6,8\n0,8",
    standingReferences: "500,800,420,1.70\n900,650,360,1.72\n1200,500,300,1.68",
  });

  assert.deepEqual(payload.floor_image_points[0], [100, 900]);
  assert.equal(payload.standing_references.length, 3);
  assert.deepEqual(payload.standing_references[1], {
    foot_px: [900, 650],
    keypoint_height_px: 360,
    height_m: 1.72,
  });
});


test("바닥점 개수가 다르면 저장 전에 거부한다", () => {
  assert.throws(
    () => buildCalibrationPayload({
      frameWidth: "1920",
      frameHeight: "1080",
      floorImagePoints: "0,0\n1,0\n1,1\n0,1",
      floorWorldPoints: "0,0\n1,0\n1,1",
      standingReferences: "1,1,100,1.7\n2,2,90,1.7\n3,3,80,1.7",
    }),
    /개수가 같아야/,
  );
});


test("API payload를 다시 편집 가능한 폼으로 복원한다", () => {
  const payload = buildCalibrationPayload({
    frameWidth: "1000",
    frameHeight: "800",
    floorImagePoints: "0,0\n1000,0\n1000,800\n0,800",
    floorWorldPoints: "0,0\n4,0\n4,3\n0,3",
    standingReferences: "500,700,500,1.7\n500,500,400,1.7\n500,300,300,1.7",
  });

  assert.deepEqual(buildCalibrationPayload(calibrationToForm(payload)), payload);
});


test("object-fit contain 영상의 레터박스를 제외하고 마우스 좌표를 영상 좌표로 바꾼다", () => {
  const rect = { left: 100, top: 50, width: 800, height: 600 };

  // 16:9 영상은 4:3 DOM 박스 안에서 세로 75px 레터박스가 생긴다.
  assert.deepEqual(
    mapClientPointToVideo({ x: 500, y: 350 }, rect, 1920, 1080),
    [960, 540],
  );
  assert.equal(
    mapClientPointToVideo({ x: 500, y: 80 }, rect, 1920, 1080),
    null,
  );
});


test("임의의 바닥점 네 개와 다섯 실측 거리로 바닥 좌표를 삼각측량한다", () => {
  const imagePoints = [
    [0, 0],
    [10, 0],
    [2, 5],
    [8, -4],
  ] as Array<[number, number]>;
  const worldPoints = floorWorldPointsFromAnchorDistances(imagePoints, {
    ab: 6,
    ac: Math.sqrt(20),
    bc: Math.sqrt(32),
    ad: Math.sqrt(34),
    bd: Math.sqrt(10),
  });

  const expected = [[0, 0], [6, 0], [2, 4], [5, -3]];
  worldPoints.forEach((point, index) => {
    assert.ok(Math.abs(point[0] - expected[index][0]) < 1e-9);
    assert.ok(Math.abs(point[1] - expected[index][1]) < 1e-9);
  });
  const distances = anchorDistancesFromWorldPoints(worldPoints);
  assert.ok(Math.abs(distances.ad - Math.sqrt(34)) < 1e-9);
  assert.ok(Math.abs(distances.bd - Math.sqrt(10)) < 1e-9);
});


test("삼각형을 만들 수 없는 기준점 거리는 저장 전에 거부한다", () => {
  assert.throws(
    () => floorWorldPointsFromAnchorDistances(
      [[0, 0], [10, 0], [2, 5], [8, 4]],
      { ab: 6, ac: 1, bc: 1, ad: 5, bd: 5 },
    ),
    /삼각형/,
  );
});


test("임의 순서 기준점은 바닥 적용 영역을 이루는 외곽선 순서로 정렬한다", () => {
  assert.deepEqual(
    convexHullPoints([[0, 0], [10, 0], [2, 5], [8, -4], [5, 0]]),
    [[0, 0], [8, -4], [10, 0], [2, 5]],
  );
});


test("화면에서 클릭한 기립자의 발 위치와 keypoint 길이를 자동 추출한다", () => {
  const empty = () => [0, 0, 0] as [number, number, number];
  const first = Array.from({ length: 17 }, empty);
  first[0] = [100, 100, 0.9];
  first[5] = [90, 200, 0.9];
  first[6] = [110, 200, 0.9];
  first[11] = [92, 300, 0.9];
  first[12] = [108, 300, 0.9];
  first[13] = [94, 400, 0.9];
  first[14] = [106, 400, 0.9];
  first[15] = [96, 500, 0.9];
  first[16] = [104, 500, 0.9];

  const second = first.map(([x, y, confidence]) => [x + 300, y, confidence] as [number, number, number]);
  const reference = selectStandingReferenceAtPoint(
    [
      { keypoints: first },
      { keypoints: second },
    ],
    [800, 337.5],
    { frameWidth: 640, frameHeight: 640 },
    { calibrationWidth: 1280, calibrationHeight: 720 },
    1.72,
  );

  assert.ok(reference);
  assert.deepEqual(reference.foot_px, [800, 562.5]);
  assert.equal(reference.height_m, 1.72);
  assert.ok(reference.keypoint_height_px > 450);
});


test("발목이 보이지 않는 사람은 기립 기준으로 저장하지 않는다", () => {
  const keypoints = Array.from(
    { length: 17 },
    (_, index) => [100, 100 + index * 10, index < 15 ? 0.9 : 0] as [number, number, number],
  );
  assert.equal(
    selectStandingReferenceAtPoint(
      [{ keypoints }],
      [100, 180],
      { frameWidth: 640, frameHeight: 640 },
      { calibrationWidth: 640, calibrationHeight: 640 },
      1.7,
    ),
    null,
  );
});
