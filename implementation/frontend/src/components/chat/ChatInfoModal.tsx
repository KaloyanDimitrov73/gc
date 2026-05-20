import { Info, X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

interface ChatInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatInfoModal({ isOpen, onClose }: ChatInfoModalProps) {
  const { t } = useLanguage();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 px-4">
      <div role="dialog" aria-modal="true" aria-labelledby="info-modal-title" className="mx-4 flex max-h-[85dvh] w-full max-w-[calc(100vw-2rem)] flex-col rounded-lg bg-white shadow-xl dark:bg-gray-800 sm:max-w-lg">
        <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-700 sm:p-6">
          <h3 id="info-modal-title" className="flex items-center gap-2 text-gray-900 dark:text-white">
            <Info className="h-5 w-5 text-teal-600" />
            {t('chat.supportedQuestionTypesTitle')}
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
        <div className="flex-1 space-y-4 overflow-y-auto p-4 text-sm text-gray-700 dark:text-gray-300 sm:p-6">
          <p>
            {t('chat.infoIntro')}
            <a
              href="https://orkg.org"
              target="_blank"
              rel="noopener noreferrer"
              className="mx-1 text-teal-600 underline dark:text-teal-400"
            >
              Open Research Knowledge Graph (ORKG)
            </a>
            {t('chat.infoFocused')} <strong>{t('chat.softwareArchitecture')}</strong> {t('chat.infoResearch')}
          </p>
          <p>{t('chat.infoCurrentSupport')}</p>
          <ul className="ml-2 list-inside list-disc space-y-1">
            <li>{t('chat.infoQ1')}</li>
            <li>{t('chat.infoQ2')}</li>
            <li>{t('chat.infoQ3')}</li>
            <li>{t('chat.infoQ4')}</li>
          </ul>
          <p>
            <strong>{t('chat.infoNote')}</strong> {t('chat.infoNoteText')}
          </p>
          <p>
            <strong>{t('chat.infoExamples')}</strong>
          </p>
          <ul className="ml-2 list-inside list-disc space-y-1">
            <li>{t('chat.infoEx1')}</li>
            <li>{t('chat.infoEx2')}</li>
            <li>{t('chat.infoEx3')}</li>
            <li>{t('chat.infoEx4')}</li>
            <li>{t('chat.infoEx5')}</li>
          </ul>
        </div>
        <div className="flex justify-end border-t border-gray-200 p-4 dark:border-gray-700 sm:p-6">
          <button
            onClick={onClose}
            className="rounded-lg bg-teal-600 px-6 py-2 text-white transition-colors hover:bg-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
          >
            {t('chat.gotIt')}
          </button>
        </div>
      </div>
    </div>
  );
}
