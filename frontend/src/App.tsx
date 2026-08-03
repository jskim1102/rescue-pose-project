import { Routes, Route } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import SettingsPage from "./pages/SettingsPage";
import { CameraRegistryProvider } from "./hooks/useCameraRegistry";

/**
 * 라우팅 셸 — 정확히 2 페이지(U4/D2 확정): / = 관제 대시보드, /settings = 설정.
 * Home 랜딩·Webcam 없음(§3 RTSP 전용). 두 페이지 모두 자체 Topbar 를 두므로
 * 글로벌 Header 는 없다.
 */
export default function App() {
  return (
    <CameraRegistryProvider>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </CameraRegistryProvider>
  );
}
