import { Download, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

interface GraphToolbarProps {
  onDownload: () => void;
  onReset: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  hasExpandedNodes: boolean;
}

export function GraphToolbar({ onDownload, onReset, onZoomIn, onZoomOut, hasExpandedNodes }: GraphToolbarProps) {
  const { t } = useLanguage();

  return (
    <>
      {/* Top-right: download + reset */}
      <div className="absolute right-2 top-2 flex flex-col gap-1 sm:right-3 sm:top-3">
        {/* address rule 2.5.2 Pointer Cancellation */}
        {/* address rule 2.5.8 Target Size (Minimum) */}
        <button
          onClick={onDownload}
          aria-label={t('graph.downloadSvg')}
          className="group relative rounded-lg border border-gray-200 bg-white p-2 shadow-sm transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:border-gray-500 dark:bg-gray-700 dark:hover:bg-gray-800"
        >
          <Download className="w-4 h-4 text-gray-600 group-hover:text-gray-900 dark:text-gray-100 dark:group-hover:text-white" />
          <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-900">
            {t('graph.downloadSvg')}
          </span>
        </button>
        <button
          onClick={onReset}
          disabled={!hasExpandedNodes}
          aria-label={t('graph.clear')}
          className="group relative rounded-lg border border-gray-200 bg-white p-2 shadow-sm transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:border-gray-500 dark:bg-gray-700 dark:hover:bg-gray-800"
        >
          <RotateCcw className="w-4 h-4 text-gray-600 group-hover:text-gray-900 dark:text-gray-100 dark:group-hover:text-white" />
          <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-900">
            {t('graph.clear')}
          </span>
        </button>
      </div>

      {/* Bottom-right: zoom in/out */}
      <div className="absolute bottom-2 right-2 flex flex-col gap-1 sm:bottom-3 sm:right-3">
        <button
          onClick={onZoomIn}
          aria-label={t('graph.zoomIn')}
          className="group relative rounded-lg border border-gray-200 bg-white p-2 shadow-sm transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:border-gray-500 dark:bg-gray-700 dark:hover:bg-gray-800"
        >
          <ZoomIn className="w-4 h-4 text-gray-600 group-hover:text-gray-900 dark:text-gray-100 dark:group-hover:text-white" />
          <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-900">
            {t('graph.zoomIn')}
          </span>
        </button>
        <button
          onClick={onZoomOut}
          aria-label={t('graph.zoomOut')}
          className="group relative rounded-lg border border-gray-200 bg-white p-2 shadow-sm transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:border-gray-500 dark:bg-gray-700 dark:hover:bg-gray-800"
        >
          <ZoomOut className="w-4 h-4 text-gray-600 group-hover:text-gray-900 dark:text-gray-100 dark:group-hover:text-white" />
          <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-900">
            {t('graph.zoomOut')}
          </span>
        </button>
      </div>
    </>
  );
}
