/**
 * 구조 필요 UI 경보 배너 (phase4, U2 = UI-only).
 *
 * needs-rescue 대상이 1명 이상일 때만 렌더(count<=0 이면 null → 자동 해제). backend RescueTracker
 * 가 posture 변경/감지 gap(expiry)로 rescueNeeded 를 내리면 WS items 갱신 → count 0 → 배너 사라짐.
 *
 * 액션 버튼(119 출동 요청 · 우선순위 전송)은 **비기능 시각 요소**다 — 외부통지(소리/푸시/웹훅/실
 * 출동)는 범위 밖(U2 확정). 실제 배선은 향후 §3 사용자 컨펌 후. onClick 없음(디자인만).
 */

const C = {
  red: "#e5484d",
  redDim: "#3a1416",
  text: "#e6edf3",
  sub: "#f3b0b2",
};

interface Props {
  count: number;
  topLabel?: string;
  topLyingSec?: number;
}

function RescueAlert({ count, topLabel, topLyingSec }: Props) {
  if (count <= 0) return null;

  return (
    <div style={styles.banner} role="alert" aria-live="assertive">
      <span style={styles.icon}>🚨</span>
      <div style={styles.body}>
        <div style={styles.title}>구조 필요 감지 — {count}명</div>
        <div style={styles.subline}>
          {topLabel
            ? `최우선 대상: ${topLabel} · 누운 지 ${Math.floor(topLyingSec ?? 0)}초 경과`
            : "즉시 현장 확인이 필요합니다"}
        </div>
      </div>
      {/* UI-only (U2) — 실제 외부 전송/출동요청 없음. 향후 §3 컨펌 후 배선. */}
      <div style={styles.actions}>
        <button type="button" style={styles.btnPrimary}>🚑 119 출동 요청</button>
        <button type="button" style={styles.btnGhost}>📨 우선순위 전송</button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  banner: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
    padding: "0.9rem 1.1rem",
    background: C.redDim,
    border: `1.5px solid ${C.red}`,
    borderRadius: "12px",
    boxShadow: `0 0 0 3px rgba(229,72,77,0.12)`,
  },
  icon: { fontSize: "1.7rem", lineHeight: 1 },
  body: { flex: 1, minWidth: 0 },
  title: { fontSize: "1.05rem", fontWeight: 800, color: C.red },
  subline: { fontSize: "0.82rem", color: C.sub, marginTop: "0.15rem" },
  actions: { display: "flex", gap: "0.5rem", flexShrink: 0 },
  btnPrimary: {
    padding: "0.5rem 0.9rem",
    borderRadius: "8px",
    border: "none",
    background: C.red,
    color: "#fff",
    fontWeight: 700,
    fontSize: "0.85rem",
    cursor: "pointer",
  },
  btnGhost: {
    padding: "0.5rem 0.9rem",
    borderRadius: "8px",
    border: `1px solid ${C.red}`,
    background: "transparent",
    color: C.text,
    fontWeight: 600,
    fontSize: "0.85rem",
    cursor: "pointer",
  },
};

export default RescueAlert;
