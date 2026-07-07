import { useCallback, useEffect, useState } from "react";
import Topbar from "../components/Topbar";
import SegmentedToggle from "../components/SegmentedToggle";
import ModelManagerModal from "../components/ModelManagerModal";
import ModelSettingsModal, { type ModelSettings } from "../components/ModelSettingsModal";
import { apiBase } from "../hooks/useApi";

/**
 * 설정 페이지 — tunnel SettingsPage(다크/레드 카메라 등록 폼 + 현황 리스트) 이식에
 * rtsp-keypoint CamerasPage 의 per-camera 추론 컨트롤(모델 선택·설정·추론 토글)을 통합.
 *
 * - 카메라 등록(RTSP) + 현황 리스트(상태/시청자/삭제) = tunnel.
 * - 현황 각 행에 [모델][설정][추론] 컨트롤 = CamerasPage. 모델은 preset-only 게이트 유지
 *   (ModelManagerModal — .pt 업로드 없음, RCE 차단). 라이브 그리드/오버레이는 관제(/) 소관.
 * - 백엔드 없으면 fetch 는 graceful(빈 리스트) — gate-2 는 백엔드 없이 렌더된다.
 */

const C = {
  bg: "#0a0d12",
  panel: "#11151c",
  panel2: "#161b24",
  border: "#232a36",
  text: "#e6edf3",
  muted: "#8b95a5",
  red: "#e5484d",
  green: "#3fb950",
};

const DEFAULT_CONF = 0.5; // 워커 conf fallback (YOLO_CONF_THRESHOLD 정본).

interface IpCam {
  id: number;
  name: string;
  rtsp_url: string;
  stream_key: string;
}
interface CamStats {
  active: boolean;
  readers: number;
}

function SettingsPage() {
  // ── 등록 폼 ──
  const [name, setName] = useState("");
  const [rtspUrl, setRtspUrl] = useState("");
  const [focus, setFocus] = useState<"name" | "url" | null>(null);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // ── 카메라 · 상태 ──
  const [cams, setCams] = useState<IpCam[]>([]);
  const [stats, setStats] = useState<Record<string, CamStats>>({});

  // ── 추론 컨트롤 (CamerasPage 차용) ──
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});
  const [confs, setConfs] = useState<Record<string, number>>({});
  const [modelsByCam, setModelsByCam] = useState<Record<string, string[] | null>>({});
  const [modelSettingsByCam, setModelSettingsByCam] = useState<
    Record<string, Record<string, ModelSettings>>
  >({});
  const [modalCamKey, setModalCamKey] = useState<string | null>(null); // 모델 관리 모달
  const [confModalCamKey, setConfModalCamKey] = useState<string | null>(null); // 모델 설정 모달

  const fetchCams = useCallback(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/ipcams`);
      if (!r.ok) return;
      setCams(await r.json());
    } catch {
      /* 백엔드 없음 — graceful */
    }
  }, []);

  useEffect(() => {
    fetchCams();
  }, [fetchCams]);

  // 연동 상태 — 2초 polling
  useEffect(() => {
    if (cams.length === 0) return;
    let cancelled = false;
    const poll = async () => {
      const entries = await Promise.all(
        cams.map(async (c) => {
          try {
            const r = await fetch(`${apiBase()}/api/ipcams/${c.stream_key}/stats`);
            return [c.stream_key, (await r.json()) as CamStats] as const;
          } catch {
            return [c.stream_key, { active: false, readers: 0 }] as const;
          }
        })
      );
      if (!cancelled) setStats(Object.fromEntries(entries));
    };
    poll();
    const id = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [cams]);

  // 카메라별 추론(enabled + conf + models) 상태 초기 fetch. 규칙: 모델을 명시 선택해야만 추론 ON.
  //   null(미설정)·[](명시 비움) 둘 다 "추론 안 함" 으로 동일 취급 — "(기본)" 자동 추론 없음.
  //   UX 정규화(ON+빈 모델 금지): 모델 없이 enabled 면 로드-타임에 enabled=false 로 교정.
  useEffect(() => {
    if (cams.length === 0) return;
    let cancelled = false;
    async function fetchAll() {
      const results = await Promise.all(
        cams.map(async (cam) => {
          try {
            const r = await fetch(`${apiBase()}/api/ipcams/${cam.stream_key}/inference`);
            const data = await r.json();
            return [cam.stream_key, data] as const;
          } catch {
            return [cam.stream_key, { enabled: false, conf_threshold: null, models: null }] as const;
          }
        })
      );
      if (cancelled) return;
      const normalized = results.map(([k, v]) => {
        const models = (v.models ?? null) as string[] | null;
        // 모델 명시 선택해야만 추론 유효 — null(미설정)/[] 둘 다 추론 OFF.
        const en = !!v.enabled && (models?.length ?? 0) > 0;
        return [k, { enabled: en, conf: v.conf_threshold ?? DEFAULT_CONF, models }] as const;
      });
      setEnabled(Object.fromEntries(normalized.map(([k, v]) => [k, v.enabled])));
      setConfs(Object.fromEntries(normalized.map(([k, v]) => [k, v.conf])));
      setModelsByCam(Object.fromEntries(normalized.map(([k, v]) => [k, v.models])));

      // 로드-타임 동기화 PUT — 모델 없이 ON 인 무효 조합 교정 (null/[] 모두 추론 OFF 로).
      await Promise.all(
        results.map(([k, v]) => {
          const models = (v.models ?? []) as string[];
          const desiredEn = !!v.enabled && models.length > 0;
          if (!!v.enabled === desiredEn) return Promise.resolve();
          return fetch(`${apiBase()}/api/ipcams/${k}/inference`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: desiredEn }),
          }).catch(() => {});
        })
      );
    }
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [cams]);

  const toggleInference = async (streamKey: string, on: boolean) => {
    setEnabled((prev) => ({ ...prev, [streamKey]: on })); // 낙관적
    try {
      const r = await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: on }),
      });
      const data = await r.json();
      setEnabled((prev) => ({ ...prev, [streamKey]: !!data.enabled }));
    } catch {
      setEnabled((prev) => ({ ...prev, [streamKey]: !on }));
    }
  };

  const handleModelsChange = async (streamKey: string, list: string[]) => {
    setModelsByCam((prev) => ({ ...prev, [streamKey]: list })); // 낙관적
    // UX 규칙: 모델이 모두 해제되면 추론도 자동 OFF (ON+빈 모델 조합 방지).
    const body: Record<string, unknown> = { models: list };
    if (list.length === 0 && (enabled[streamKey] ?? false)) {
      setEnabled((prev) => ({ ...prev, [streamKey]: false }));
      body.enabled = false;
    }
    try {
      await fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      /* ignore */
    }
  };

  // 모델 설정 변경 — 클라 settings 갱신 + per-source conf_threshold(워커 conf) 동기화 PUT.
  const handleSettingsChange = (streamKey: string, next: Record<string, ModelSettings>) => {
    setModelSettingsByCam((prev) => ({ ...prev, [streamKey]: next }));
    const models = modelsByCam[streamKey] ?? [];
    const confVals = models.map((m) => next[m]?.conf ?? DEFAULT_CONF);
    const workerConf = confVals.length ? Math.min(...confVals) : DEFAULT_CONF;
    setConfs((prev) => ({ ...prev, [streamKey]: workerConf }));
    fetch(`${apiBase()}/api/ipcams/${streamKey}/inference`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conf_threshold: workerConf }),
    }).catch(() => {});
  };

  const handleDelete = async (cam: IpCam) => {
    if (!window.confirm(`${cam.name} 삭제?`)) return;
    try {
      await fetch(`${apiBase()}/api/ipcams/${cam.id}`, { method: "DELETE" });
      fetchCams();
    } catch {
      setMsg({ kind: "err", text: "삭제 실패 (백엔드 확인)" });
    }
  };

  const handleAdd = async () => {
    if (submitting) return; // 중복 제출 방지
    if (!rtspUrl.trim()) {
      setMsg({ kind: "err", text: "RTSP 주소를 입력하세요" });
      return;
    }
    const camName = name.trim() || "카메라";
    setSubmitting(true);
    try {
      const r = await fetch(`${apiBase()}/api/ipcams`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: camName, rtsp_url: rtspUrl.trim() }),
      });
      if (r.status === 409) {
        const body = await r.json().catch(() => ({}));
        setMsg({ kind: "err", text: body.detail ?? "등록 한도 초과" });
        return;
      }
      if (!r.ok) throw new Error(String(r.status));
      setName("");
      setRtspUrl("");
      setMsg({ kind: "ok", text: `등록됨: ${camName}` });
      fetchCams();
    } catch {
      setMsg({ kind: "err", text: "등록 실패 (백엔드 확인)" });
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle = (k: "name" | "url"): React.CSSProperties => ({
    ...s.input,
    borderColor: focus === k ? C.red : C.border,
  });

  return (
    <div style={s.root}>
      {/* 상단 바 (공용) */}
      <Topbar active="settings" />

      <div style={s.body}>
        {/* 카메라 등록 카드 */}
        <section style={s.panel}>
          <div style={s.panelHeader}>
            <span style={s.panelTitle}>📹 카메라 등록</span>
            <span style={s.panelTitleSub}>CAMERA REGISTER</span>
          </div>

          <div style={s.formRow}>
            <div style={s.field}>
              <input
                style={inputStyle("name")}
                placeholder="카메라 이름 (ID)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onFocus={() => setFocus("name")}
                onBlur={() => setFocus(null)}
              />
            </div>
            <div style={{ ...s.field, flex: 1 }}>
              <input
                style={inputStyle("url")}
                placeholder="rtsp://192.168.0.100:554/stream"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                onFocus={() => setFocus("url")}
                onBlur={() => setFocus(null)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              />
            </div>
            <button
              style={{ ...s.addBtn, ...(submitting ? { opacity: 0.6, cursor: "default" } : {}) }}
              onClick={handleAdd}
              disabled={submitting}
            >
              {submitting ? "등록 중…" : "추가"}
            </button>
          </div>

          {msg && (
            <div style={{ ...s.msg, color: msg.kind === "ok" ? C.green : C.red }}>{msg.text}</div>
          )}
        </section>

        {/* 등록된 카메라 현황 + 추론 컨트롤 */}
        <section style={s.panel}>
          <div style={s.panelHeader}>
            <span style={s.panelTitle}>🎥 등록된 카메라 <span style={s.countTag}>{cams.length}</span></span>
            <span style={s.panelTitleSub}>REGISTERED CAMERAS</span>
          </div>

          {cams.length === 0 ? (
            <div style={s.empty}>등록된 카메라 없음 — 위에서 RTSP 추가</div>
          ) : (
            <div style={s.list}>
              {/* 헤더 행 */}
              <div style={{ ...s.row, ...s.rowHead }}>
                <span style={s.colStatus}>상태</span>
                <span style={s.colName}>이름</span>
                <span style={s.colUrl}>RTSP 주소</span>
                <span style={s.colCtrl}>추론</span>
                <span style={s.colAct} />
              </div>
              {cams.map((c) => {
                const st = stats[c.stream_key];
                const online = !!st?.active;
                const models = modelsByCam[c.stream_key]; // string[] | null | undefined
                const sel = models ?? [];
                const hasModels = sel.length > 0;
                // 추론 유효 = 명시 모델 선택 필요. 모델 없으면 토글 disable.
                const canInfer = hasModels;
                const camEnabled = enabled[c.stream_key] ?? false;
                const modelText = !hasModels
                  ? "모델 없음"
                  : sel.length === 1
                  ? sel[0]
                  : `${sel[0]} +${sel.length - 1}`;
                return (
                  <div key={c.id} style={s.row}>
                    <span style={s.colStatus}>
                      <span style={{ ...s.dot, background: online ? C.green : C.muted }} />
                      <span style={{ color: online ? C.green : C.muted, fontSize: "0.72rem", fontWeight: 700 }}>
                        {online ? "LIVE" : "대기"}
                      </span>
                    </span>
                    <span style={s.colName}>{c.name}</span>
                    <span style={{ ...s.colUrl, color: C.muted, fontFamily: "monospace", fontSize: "0.78rem" }}>
                      {c.rtsp_url}
                    </span>
                    <span style={s.colCtrl}>
                      {/* 모델 선택 → 모델 설정 → 추론 토글 (preset-only 게이트) */}
                      <button style={s.ctrlBtn} title={modelText} onClick={() => setModalCamKey(c.stream_key)}>
                        모델{hasModels ? ` (${sel.length})` : ""}
                      </button>
                      <button
                        style={{ ...s.ctrlBtn, ...(hasModels ? {} : s.ctrlBtnDisabled) }}
                        disabled={!hasModels}
                        onClick={() => setConfModalCamKey(c.stream_key)}
                      >
                        설정
                      </button>
                      <SegmentedToggle
                        enabled={camEnabled}
                        onChange={(on) => toggleInference(c.stream_key, on)}
                        disabled={!canInfer}
                      />
                    </span>
                    <span style={s.colAct}>
                      <button style={s.delBtn} onClick={() => handleDelete(c)}>삭제</button>
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>

      {/* 모델 관리 모달 (preset 선택 — .pt 업로드 없음, RCE 차단) */}
      {modalCamKey !== null && (
        <ModelManagerModal
          open={modalCamKey !== null}
          onClose={() => setModalCamKey(null)}
          cameraName={cams.find((c) => c.stream_key === modalCamKey)?.name ?? "카메라"}
          selected={modelsByCam[modalCamKey] ?? []}
          onSelectedChange={(list) => handleModelsChange(modalCamKey, list)}
        />
      )}

      {/* 모델 설정 모달 (사람 단위 conf 슬라이더) */}
      {confModalCamKey !== null && (
        <ModelSettingsModal
          open={confModalCamKey !== null}
          onClose={() => setConfModalCamKey(null)}
          cameraName={cams.find((c) => c.stream_key === confModalCamKey)?.name ?? "카메라"}
          fallbackConf={confs[confModalCamKey] ?? DEFAULT_CONF}
          selectedModels={modelsByCam[confModalCamKey] ?? []}
          settings={modelSettingsByCam[confModalCamKey] ?? {}}
          onSettingsChange={(next) => handleSettingsChange(confModalCamKey, next)}
        />
      )}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  root: { minHeight: "100vh", background: C.bg, color: C.text },

  body: { padding: "1rem", display: "flex", flexDirection: "column", gap: "1rem" },

  panel: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: "12px", padding: "1rem" },
  panelHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" },
  panelTitle: { fontSize: "0.95rem", fontWeight: 700 },
  panelTitleSub: { fontSize: "0.72rem", color: C.muted, letterSpacing: "0.08em" },

  formRow: { display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" },
  field: { display: "flex", flexDirection: "column" },
  input: {
    padding: "0.6rem 0.75rem", borderRadius: "8px", border: `1px solid ${C.border}`,
    background: C.panel2, color: C.text, fontSize: "0.9rem", outline: "none",
    minWidth: "150px", transition: "border-color 0.15s",
  },
  addBtn: {
    padding: "0.6rem 1.4rem", borderRadius: "8px", border: "none",
    background: C.red, color: "#fff", fontWeight: 700, fontSize: "0.9rem", cursor: "pointer",
  },
  msg: { marginTop: "0.7rem", fontSize: "0.8rem" },

  countTag: {
    display: "inline-block", marginLeft: "0.4rem", padding: "0.05rem 0.5rem",
    background: "rgba(229,72,77,0.15)", color: C.red, borderRadius: "10px",
    fontSize: "0.72rem", fontWeight: 700,
  },
  empty: { color: C.muted, fontSize: "0.85rem", padding: "1rem 0", textAlign: "center" },
  list: { display: "flex", flexDirection: "column" },
  row: {
    display: "grid",
    gridTemplateColumns: "88px 130px 1fr 300px 60px",
    alignItems: "center", gap: "0.75rem",
    padding: "0.6rem 0.5rem", borderBottom: `1px solid ${C.border}`,
    fontSize: "0.85rem",
  },
  rowHead: { color: C.muted, fontSize: "0.72rem", fontWeight: 600 },
  colStatus: { display: "flex", alignItems: "center", gap: "0.35rem" },
  colName: { fontWeight: 600 },
  colUrl: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  colCtrl: { display: "flex", alignItems: "center", gap: "0.4rem", flexWrap: "wrap" },
  colAct: { textAlign: "right" },
  dot: { width: "8px", height: "8px", borderRadius: "50%", display: "inline-block" },
  ctrlBtn: {
    padding: "0.35rem 0.7rem", borderRadius: "6px", border: `1px solid ${C.border}`,
    background: C.panel2, color: C.text, fontSize: "0.78rem", fontWeight: 600, cursor: "pointer",
  },
  ctrlBtnDisabled: { opacity: 0.45, cursor: "not-allowed" },
  delBtn: {
    padding: "0.3rem 0.7rem", borderRadius: "6px", border: `1px solid ${C.red}`,
    background: "transparent", color: C.red, fontSize: "0.78rem", fontWeight: 600, cursor: "pointer",
  },
};

export default SettingsPage;
