import { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import { apiClient } from '../services/api';

interface HealthContextValue {
  backendStatus: string;
}

const HealthContext = createContext<HealthContextValue | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const [backendStatus, setBackendStatus] = useState<string>('checking');
  const initCalled = useRef(false);

  useEffect(() => {
    if (initCalled.current) return;
    initCalled.current = true;

    const checkBackendHealth = async () => {
      try {
        const health = await apiClient.checkHealth();

        if (health.hublinkAvailable) {
          setBackendStatus('hublink');
        } else {
          setBackendStatus('initializing');
          apiClient.initHublink()
            .then((res) => setBackendStatus(res.hublinkAvailable ? 'hublink' : 'mock'))
            .catch(() => setBackendStatus('mock'));
        }
      } catch (error) {
        console.error('Backend health check failed:', error);
        setBackendStatus('offline');
      }
    };

    checkBackendHealth();
  }, []);

  return (
    <HealthContext.Provider value={{ backendStatus }}>
      {children}
    </HealthContext.Provider>
  );
}

export function useHealth(): HealthContextValue {
  const context = useContext(HealthContext);
  if (!context) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
}