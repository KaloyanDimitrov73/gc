import { Info } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

interface ChatHeaderProps {
  onOpenInfo: () => void;
}

export function ChatHeader({ onOpenInfo }: ChatHeaderProps) {
  const { t } = useLanguage();

  return (
    <div className="shrink-0 border-b border-gray-200 px-4 py-2 dark:border-gray-800 md:px-5 lg:px-6">
      <div className="inline-flex items-center gap-2">
            {/* address rule 2.5.2 Pointer Cancellation */}
            {/* address rule 1.4.11 Non-text Contrast */}
            <button
              onClick={onOpenInfo}
              className="group relative inline-flex flex-shrink-0 items-center rounded p-1 text-teal-600 transition-colors hover:text-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-teal-400 dark:hover:text-teal-300"
              aria-label={t('chat.supportedQuestionTypes')}
            >
              <Info className="h-5 w-5" />
              {/* address rule 1.4.13 Content on Hover or Focus */}
              <span aria-hidden="true" className="pointer-events-none absolute top-full left-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
                {t('chat.supportedQuestionTypes')}
              </span>
            </button>
            <p className="text-xs text-gray-600 dark:text-gray-400 sm:text-sm">
              {t('chat.demoDescription')}
            </p>
      </div>
    </div>
  );
}
