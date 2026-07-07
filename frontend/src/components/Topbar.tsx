import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";

/**
 * 모든 페이지 공용 상단 바 (tunnel Topbar 이식 · rescue-pose 리브랜드).
 * 3-컬럼 그리드(1fr auto 1fr)로 브랜드(좌) · nav(정중앙) · 우측 슬롯 을 고정 —
 * 우측 콘텐츠가 달라도 nav 위치/브랜드 크기가 페이지마다 동일하다.
 */

const C = {
  panel: "#11151c",
  border: "#232a36",
  muted: "#8b95a5",
  red: "#e5484d",
};

type Tab = "monitor" | "settings";

function Topbar({ active, right }: { active: Tab; right?: ReactNode }) {
  const navigate = useNavigate();
  return (
    <header style={s.topbar}>
      {/* 좌 — 브랜드 (rescue-pose) */}
      <div style={s.brandWrap}>
        <span style={s.brandIcon}>🚨</span>
        <div>
          <div style={s.brandTitle}>RESCUE&nbsp;POSE</div>
          <div style={s.brandSub}>자세 기반 응급 구조 판단 · 실시간 관제</div>
        </div>
      </div>

      {/* 중앙 — nav */}
      <nav style={s.nav}>
        <span
          style={{ ...s.navItem, ...(active === "monitor" ? s.navActive : null) }}
          onClick={() => active !== "monitor" && navigate("/")}
        >
          🛡 관제
        </span>
        <span
          style={{ ...s.navItem, ...(active === "settings" ? s.navActive : null) }}
          onClick={() => active !== "settings" && navigate("/settings")}
        >
          ⚙ 설정
        </span>
      </nav>

      {/* 우 — 페이지별 슬롯 */}
      <div style={s.right}>{right}</div>
    </header>
  );
}

const s: Record<string, React.CSSProperties> = {
  topbar: {
    display: "grid",
    gridTemplateColumns: "1fr auto 1fr",
    alignItems: "center",
    minHeight: "var(--rp-topbar-h, 56px)",
    padding: "var(--rp-topbar-pad-y) var(--rp-topbar-pad-x)",
    background: C.panel,
    borderBottom: `1px solid ${C.border}`,
  },
  brandWrap: { display: "flex", alignItems: "center", gap: "0.65rem", justifySelf: "start", minWidth: 0 },
  brandIcon: { fontSize: "1.32rem" },
  brandTitle: { fontSize: "1.05rem", fontWeight: 700, letterSpacing: "0.02em" },
  brandSub: { fontSize: "0.68rem", color: C.muted },
  nav: { display: "flex", gap: "0.5rem", justifySelf: "center" },
  navItem: { padding: "0.4rem 0.9rem", borderRadius: "8px", fontSize: "0.88rem", color: C.muted, cursor: "pointer", userSelect: "none" },
  navActive: { color: C.red, background: "rgba(229,72,77,0.12)", border: `1px solid ${C.red}` },
  right: { display: "flex", alignItems: "center", gap: "0.8rem", justifySelf: "end" },
};

export default Topbar;
