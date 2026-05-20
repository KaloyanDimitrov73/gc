import { Search, X, ChevronUp, ChevronDown } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

interface GraphSearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onClear: () => void;
  matchCount?: number;
  currentMatchIndex?: number;
  onNext?: () => void;
  onPrev?: () => void;
}

export function GraphSearchBar({ value, onChange, onClear, matchCount = 0, currentMatchIndex = 0, onNext, onPrev }: GraphSearchBarProps) {
  const { t } = useLanguage();

  const showNav = value.length > 0 && matchCount > 0;

  return (
    <div className="relative min-w-0 w-full">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 dark:text-gray-500" />
      {/* address rule 3.3.2 Labels or Instructions */}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={t('graph.searchPlaceholder')}
        aria-label={t('graph.searchAriaLabel')}
        className={`w-full pl-9 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent ${showNav ? 'pr-28' : 'pr-8'}`}
      />
      {value && (
        <>
          {showNav && (
            <div className="absolute right-8 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
              <span className="text-xs text-gray-400 dark:text-gray-500 tabular-nums px-1">
                {currentMatchIndex + 1}/{matchCount}
              </span>
              <button
                onClick={onPrev}
                title={t('graph.prevMatch')}
                aria-label={t('graph.prevMatch')}
                className="rounded p-0.5 transition-colors hover:bg-gray-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-700"
              >
                <ChevronUp className="w-3.5 h-3.5 text-gray-500 dark:text-gray-400" />
              </button>
              <button
                onClick={onNext}
                title={t('graph.nextMatch')}
                aria-label={t('graph.nextMatch')}
                className="rounded p-0.5 transition-colors hover:bg-gray-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-700"
              >
                <ChevronDown className="w-3.5 h-3.5 text-gray-500 dark:text-gray-400" />
              </button>
            </div>
          )}
          {/* address rule 2.5.2 Pointer Cancellation */}
          {/* address rule 1.4.11 Non-text Contrast */}
          <button
            onClick={onClear}
            title={t('graph.clearSearch')}
            aria-label={t('graph.clearSearch')}
            className="group absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 transition-colors hover:bg-gray-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-700"
          >
            <X className="w-3.5 h-3.5 text-gray-400 dark:text-gray-500" />
            {/* address rule 1.4.13 Content on Hover or Focus */}
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
              {t('graph.clearSearch')}
            </span>
          </button>
        </>
      )}
    </div>
  );
}
