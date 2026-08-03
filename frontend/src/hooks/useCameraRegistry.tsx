import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useLocation } from "react-router-dom";
import { apiBase } from "./useApi";

export interface RegisteredCamera {
  id: number;
  name: string;
  rtsp_url: string;
  stream_key: string;
}

export type CameraListStatus = "loading" | "ready" | "error";

interface CameraRegistry {
  cameras: RegisteredCamera[];
  status: CameraListStatus;
  refreshCameras: () => Promise<boolean>;
}

const CameraRegistryContext = createContext<CameraRegistry | null>(null);

export function CameraRegistryProvider({ children }: { children: ReactNode }) {
  const [cams, setCams] = useState<RegisteredCamera[]>([]);
  const [status, setStatus] = useState<CameraListStatus>("loading");
  const { pathname } = useLocation();

  const refreshCameras = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase()}/api/ipcams`);
      if (!response.ok) {
        throw new Error(`camera list request failed: ${response.status}`);
      }
      const data: unknown = await response.json();
      if (!Array.isArray(data)) {
        throw new Error("camera list response is not an array");
      }
      setCams(data as RegisteredCamera[]);
      setStatus("ready");
      return true;
    } catch {
      // 초기 실패만 오류로 표시한다. 갱신 실패는 마지막 정상 목록을 그대로 유지한다.
      setStatus((status) => (status === "ready" ? status : "error"));
      return false;
    }
  }, []);

  useEffect(() => {
    void refreshCameras();
  }, [refreshCameras, pathname]);

  const value = useMemo(
    () => ({ cameras: cams, status, refreshCameras }),
    [cams, status, refreshCameras],
  );

  return (
    <CameraRegistryContext.Provider value={value}>
      {children}
    </CameraRegistryContext.Provider>
  );
}

export function useCameraRegistry(): CameraRegistry {
  const registry = useContext(CameraRegistryContext);
  if (!registry) {
    throw new Error("useCameraRegistry must be used within CameraRegistryProvider");
  }
  return registry;
}
