import { Brain, Database, Layers, Network, Settings, X, Zap } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useSettings } from '../../contexts/SettingsContext';

interface ChatSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatSettingsModal({ isOpen, onClose }: ChatSettingsModalProps) {
  const { t } = useLanguage();
  const { retrievalMode, setRetrievalMode, llmModel, setLlmModel, numberOfHubs, setNumberOfHubs, useDirectFinalAnswer, setUseDirectFinalAnswer, availableModels, modelsLoading } =
    useSettings();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 px-4">
      <div role="dialog" aria-modal="true" aria-labelledby="settings-modal-title" className="mx-4 flex max-h-[85dvh] w-full max-w-[calc(100vw-2rem)] flex-col rounded-lg bg-white shadow-xl dark:bg-gray-800 sm:max-w-xl">
        <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-700 sm:p-6">
          <h3 id="settings-modal-title" className="flex items-center gap-2 text-gray-900 dark:text-white">
            <Settings className="h-5 w-5 text-teal-600" />
            {t('chat.configSettings')}
          </h3>
          <button
            onClick={onClose}
            className="group relative rounded-lg p-1 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-700"
            aria-label={t('chat.close')}
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
              {t('chat.close')}
            </span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6">
          <div className="space-y-6">
            {/* address rule 3.3.2 Labels or Instructions */}
            <fieldset>
              <legend className="mb-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <Network className="h-4 w-4" />
                {t('chat.retrievalStrategy')}
              </legend>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {/* address rule 4.1.2 Name, Role, Value */}
                <button
                  onClick={() => setRetrievalMode('direct')}
                  aria-pressed={retrievalMode === 'direct'}
                  disabled
                  className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                    retrievalMode === 'direct'
                      ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                      : 'border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <Zap
                    className={`h-4 w-4 flex-shrink-0 ${
                      retrievalMode === 'direct' ? 'text-teal-600' : 'text-gray-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1 text-left">
                    <div
                      className={`text-sm ${
                        retrievalMode === 'direct'
                          ? 'text-teal-700 dark:text-teal-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {t('chat.direct')}
                    </div>
                    <div className="truncate text-xs text-gray-500 dark:text-gray-400">
                      {t('chat.vectorANN')}
                    </div>
                  </div>
                  {/* address rule 1.4.1 Use of Color */}
                  {retrievalMode === 'direct' && (
                    <div className="h-2 w-2 flex-shrink-0 rounded-full bg-teal-600"></div>
                  )}
                </button>
                <button
                  onClick={() => setRetrievalMode('graph')}
                  aria-pressed={retrievalMode === 'graph'}
                  disabled
                  className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                    retrievalMode === 'graph'
                      ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                      : 'border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <Network
                    className={`h-4 w-4 flex-shrink-0 ${
                      retrievalMode === 'graph' ? 'text-teal-600' : 'text-gray-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1 text-left">
                    <div
                      className={`text-sm ${
                        retrievalMode === 'graph'
                          ? 'text-teal-700 dark:text-teal-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {t('chat.graph')}
                    </div>
                    <div className="truncate text-xs text-gray-500 dark:text-gray-400">
                      {t('chat.traversal')}
                    </div>
                  </div>
                  {retrievalMode === 'graph' && (
                    <div className="h-2 w-2 flex-shrink-0 rounded-full bg-teal-600"></div>
                  )}
                </button>
              </div>
            </fieldset>

            <fieldset>
              <legend className="mb-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <Brain className="h-4 w-4" />
                {t('chat.llmModel')}
              </legend>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {modelsLoading ? (
                  <p className="col-span-2 text-sm text-gray-500 dark:text-gray-400">{t('chat.loadingModels') ?? 'Loading models…'}</p>
                ) : (
                  availableModels.map(({ model, provider }) => (
                    <button
                      key={model}
                      onClick={() => setLlmModel(model)}
                      aria-pressed={llmModel === model}
                      disabled
                      className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                        llmModel === model
                          ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                          : 'border-gray-200 dark:border-gray-600'
                      }`}
                    >
                      <Brain
                        className={`h-4 w-4 flex-shrink-0 ${
                          llmModel === model ? 'text-teal-600' : 'text-gray-400'
                        }`}
                      />
                      <div className="min-w-0 flex-1 text-left">
                        <div
                          className={`text-sm ${
                            llmModel === model
                              ? 'text-teal-700 dark:text-teal-400'
                              : 'text-gray-700 dark:text-gray-300'
                          }`}
                        >
                          {model}
                        </div>
                        <div className="truncate text-xs text-gray-500 dark:text-gray-400">
                          {provider}
                        </div>
                      </div>
                      {llmModel === model && (
                        <div className="h-2 w-2 flex-shrink-0 rounded-full bg-teal-600"></div>
                      )}
                    </button>
                  ))
                )}
              </div>
            </fieldset>

            <fieldset>
              <legend className="mb-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <Database className="h-4 w-4" />
                {t('chat.numArtifacts')}
              </legend>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {([10, 20, 30] as (10 | 20 | 30)[]).map((hubs) => (
                  <button
                    key={hubs}
                    onClick={() => setNumberOfHubs(hubs)}
                    aria-pressed={numberOfHubs === hubs}
                    disabled
                    className={`flex items-center justify-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                      numberOfHubs === hubs
                        ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                        : 'border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    <span
                      className={`text-sm ${
                        numberOfHubs === hubs
                          ? 'text-teal-700 dark:text-teal-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {hubs}
                    </span>
                    {numberOfHubs === hubs && (
                      <div className="h-2 w-2 rounded-full bg-teal-600"></div>
                    )}
                  </button>
                ))}
              </div>
            </fieldset>

            <fieldset>
              <legend className="mb-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <Layers className="h-4 w-4" />
                {t('chat.answerMode')}
              </legend>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <button
                  onClick={() => setUseDirectFinalAnswer(false)}
                  aria-pressed={!useDirectFinalAnswer}
                  className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors ${
                    !useDirectFinalAnswer
                      ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                      : 'border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <Layers
                    className={`h-4 w-4 flex-shrink-0 ${
                      !useDirectFinalAnswer ? 'text-teal-600' : 'text-gray-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1 text-left">
                    <div
                      className={`text-sm ${
                        !useDirectFinalAnswer
                          ? 'text-teal-700 dark:text-teal-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {t('chat.answerModePartial')}
                    </div>
                    <div className="truncate text-xs text-gray-500 dark:text-gray-400">
                      {t('chat.answerModePartialDesc')}
                    </div>
                  </div>
                  {!useDirectFinalAnswer && (
                    <div className="h-2 w-2 flex-shrink-0 rounded-full bg-teal-600"></div>
                  )}
                </button>
                <button
                  onClick={() => setUseDirectFinalAnswer(true)}
                  aria-pressed={useDirectFinalAnswer}
                  className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2.5 transition-colors ${
                    useDirectFinalAnswer
                      ? 'border-teal-600 bg-teal-50 dark:bg-teal-950'
                      : 'border-gray-200 dark:border-gray-600'
                  }`}
                >
                  <Zap
                    className={`h-4 w-4 flex-shrink-0 ${
                      useDirectFinalAnswer ? 'text-teal-600' : 'text-gray-400'
                    }`}
                  />
                  <div className="min-w-0 flex-1 text-left">
                    <div
                      className={`text-sm ${
                        useDirectFinalAnswer
                          ? 'text-teal-700 dark:text-teal-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {t('chat.answerModeDirect')}
                    </div>
                    <div className="truncate text-xs text-gray-500 dark:text-gray-400">
                      {t('chat.answerModeDirectDesc')}
                    </div>
                  </div>
                  {useDirectFinalAnswer && (
                    <div className="h-2 w-2 flex-shrink-0 rounded-full bg-teal-600"></div>
                  )}
                </button>
              </div>
            </fieldset>
          </div>
        </div>

        <div className="flex justify-end border-t border-gray-200 p-4 dark:border-gray-700 sm:p-6">
          <button
            onClick={onClose}
            className="rounded-lg bg-teal-600 px-6 py-2 text-white transition-colors hover:bg-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
          >
            {t('chat.done')}
          </button>
        </div>
      </div>
    </div>
  );
}
