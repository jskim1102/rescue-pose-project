import assert from "node:assert/strict";
import test from "node:test";

import {
  anchorDistancesFromWorldPoints,
  buildCalibrationPayload,
  calibrationToForm,
  convexHullPoints,
  floorWorldPointsFromAnchorDistances,
  mapClientPointToVideo,
  parsePostureCalibrationDraft,
  postureCalibrationDraftStorageKey,
  standingReferenceFromSinglePerson,
  type PostureCalibrationDraft,
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


test("화면에 한 명만 검출되면 그 사람의 발 위치와 keypoint 길이를 자동 추출한다", () => {
  const empty = () => [0, 0, 0] as [number, number, number];
  const keypoints = Array.from({ length: 17 }, empty);
  keypoints[0] = [100, 100, 0.9];
  keypoints[5] = [90, 200, 0.9];
  keypoints[6] = [110, 200, 0.9];
  keypoints[11] = [92, 300, 0.9];
  keypoints[12] = [108, 300, 0.9];
  keypoints[13] = [94, 400, 0.9];
  keypoints[14] = [106, 400, 0.9];
  keypoints[15] = [96, 500, 0.9];
  keypoints[16] = [104, 500, 0.9];

  const reference = standingReferenceFromSinglePerson(
    [{ keypoints }],
    { frameWidth: 640, frameHeight: 640 },
    { calibrationWidth: 1280, calibrationHeight: 720 },
    1.72,
  );

  assert.ok(reference);
  assert.deepEqual(reference.foot_px, [200, 562.5]);
  assert.equal(reference.height_m, 1.72);
  assert.ok(reference.keypoint_height_px > 450);
});


test("검출 인원이 0명 또는 2명 이상이면 기립 기준 대상을 선택하지 않는다", () => {
  const keypoints = Array.from(
    { length: 17 },
    (_, index) => [100, 100 + index * 10, 0.9] as [number, number, number],
  );
  const frame = { frameWidth: 640, frameHeight: 640 };
  const calibration = { calibrationWidth: 640, calibrationHeight: 640 };

  assert.equal(standingReferenceFromSinglePerson([], frame, calibration, 1.7), null);
  assert.equal(
    standingReferenceFromSinglePerson(
      [{ keypoints }, { keypoints }],
      frame,
      calibration,
      1.7,
    ),
    null,
  );
});


test("한 명이 검출돼도 발목이 보이지 않으면 기립 기준으로 저장하지 않는다", () => {
  const keypoints = Array.from(
    { length: 17 },
    (_, index) => [100, 100 + index * 10, index < 15 ? 0.9 : 0] as [number, number, number],
  );
  assert.equal(
    standingReferenceFromSinglePerson(
      [{ keypoints }],
      { frameWidth: 640, frameHeight: 640 },
      { calibrationWidth: 640, calibrationHeight: 640 },
      1.7,
    ),
    null,
  );
});


test("카메라별 보정 초안을 브라우저 저장값에서 안전하게 복원한다", () => {
  const draft: PostureCalibrationDraft = {
    version: 1,
    step: "standing",
    frameSize: { width: 1920, height: 1080 },
    floorPoints: [[120, 920], [420, 520], [1500, 520], [1800, 920]],
    anchorDistanceInputs: { ab: "3.00", ac: "6.00", bc: "5.20", ad: "7.00", bd: "4.50" },
    personHeightM: "1.77",
    standingReferences: [
      { foot_px: [800, 820], keypoint_height_px: 420, height_m: 1.77 },
    ],
  };

  assert.deepEqual(parsePostureCalibrationDraft(JSON.stringify(draft)), draft);
  assert.notEqual(postureCalibrationDraftStorageKey(1), postureCalibrationDraftStorageKey(2));
});


test("손상되거나 지원하지 않는 보정 초안은 복원하지 않는다", () => {
  assert.equal(parsePostureCalibrationDraft("not-json"), null);
  assert.equal(parsePostureCalibrationDraft(JSON.stringify({ version: 2 })), null);
  assert.equal(
    parsePostureCalibrationDraft(JSON.stringify({
      version: 1,
      step: "floor",
      frameSize: { width: 1920, height: 1080 },
      floorPoints: [[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]],
      anchorDistanceInputs: { ab: "1", ac: "1", bc: "1", ad: "1", bd: "1" },
      personHeightM: "1.70",
      standingReferences: [],
    })),
    null,
  );
});
