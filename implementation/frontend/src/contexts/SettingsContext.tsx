import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { RetrievalMode, LLMModel, LlmModelInfo, NumberOfHubs } from '../types';
import { apiClient } from '../services/api';

interface SettingsContextValue {
  retrievalMode: RetrievalMode;
  setRetrievalMode: (mode: RetrievalMode) => void;
  llmModel: LLMModel;
  setLlmModel: (model: LLMModel) => void;
  numberOfHubs: NumberOfHubs;
  setNumberOfHubs: (hubs: NumberOfHubs) => void;
  useDirectFinalAnswer: boolean;
  setUseDirectFinalAnswer: (value: boolean) => void;
  availableModels: LlmModelInfo[];
  modelsLoading: boolean;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('direct');
  const [llmModel, setLlmModel] = useState<LLMModel>('o3-mini');
  const [numberOfHubs, setNumberOfHubs] = useState<NumberOfHubs>(10);
  const [useDirectFinalAnswer, setUseDirectFinalAnswer] = useState<boolean>(true);
  const [availableModels, setAvailableModels] = useState<LlmModelInfo[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    apiClient.getLlmModels()
      .then((models) => {
        setAvailableModels(models);
        if (models.length > 0) {
          setLlmModel(models[0].model);
        }
      })
      .catch(() => {
        // Keep the hardcoded default if the endpoint is unreachable
      })
      .finally(() => setModelsLoading(false));
  }, []);

  return (
    <SettingsContext.Provider
      value={{
        retrievalMode,
        setRetrievalMode,
        llmModel,
        setLlmModel,
        numberOfHubs,
        setNumberOfHubs,
        useDirectFinalAnswer,
        setUseDirectFinalAnswer,
        availableModels,
        modelsLoading,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}
