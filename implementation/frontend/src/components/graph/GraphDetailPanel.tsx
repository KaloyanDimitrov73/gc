import { X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { LITERAL_NODE_COLOR, RESOURCE_NODE_COLOR, DETAIL_PANEL_LAYOUT } from './utils/graphConstants';
import type { NodeSelection } from './hooks/useGraphNodeExpansion';

interface GraphDetailPanelProps {
  clickedNode: NodeSelection | null;
  clickedEdge: { from: string; to: string; relation: string } | null;
  isNodeExpanding: (id: string) => boolean;
  onClose: () => void;
}

export function GraphDetailPanel({ clickedNode, clickedEdge, isNodeExpanding, onClose }: GraphDetailPanelProps) {
  const { t } = useLanguage();

  if (!clickedNode && !clickedEdge) return null;

  if (DETAIL_PANEL_LAYOUT === 'overlay') {
    return (
      <div className="absolute inset-x-3 bottom-3 z-40 max-h-[45%] overflow-y-auto rounded-2xl border border-gray-200 bg-white/95 p-3 shadow-xl backdrop-blur dark:border-gray-700 dark:bg-gray-900/95 sm:inset-x-auto sm:bottom-4 sm:left-4 sm:w-[min(280px,calc(100%-2rem))] sm:max-h-none">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            {clickedNode ? (
              <>
                <div className="flex items-center gap-2">
                  <div
                    className="h-3 w-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: clickedNode.type === 'literal' ? LITERAL_NODE_COLOR : RESOURCE_NODE_COLOR }}
                  />
                  <div
                    className="text-xs font-medium uppercase tracking-wide"
                    style={{ color: clickedNode.type === 'literal' ? LITERAL_NODE_COLOR : RESOURCE_NODE_COLOR }}
                  >
                    {clickedNode.type}
                  </div>
                </div>
                <div className="mt-1.5 text-base font-medium leading-snug text-gray-900 dark:text-white break-words">
                  {clickedNode.label}
                </div>
                {isNodeExpanding(clickedNode.id) && (
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">Loading neighbors...</div>
                )}
              </>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-teal-600 flex-shrink-0" />
                  <div className="text-xs font-medium uppercase tracking-wide text-teal-700 dark:text-teal-300">
                    {t('graph.edge')}
                  </div>
                </div>
                <div className="mt-1.5 text-base font-medium leading-snug text-gray-900 dark:text-white break-words">
                  {clickedEdge?.relation}
                </div>
              </>
            )}
          </div>
          <button
            onClick={onClose}
            className="group relative rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-800 dark:hover:text-gray-200"
            aria-label={t('graph.closeDetails')}
          >
            <X className="h-5 w-5" />
            <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
              {t('graph.closeDetails')}
            </span>
          </button>
        </div>

        {clickedNode ? (
          <div className="mt-3 inline-flex min-w-0 max-w-full items-center overflow-hidden rounded-xl border border-gray-200 dark:border-gray-700">
            <span className="bg-gray-50 px-2.5 py-1.5 text-xs font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-300">
              ID
            </span>
            <span className="px-2.5 py-1.5 text-xs text-gray-900 dark:text-white break-all">
              {clickedNode.id}
            </span>
          </div>
        ) : clickedEdge ? (
          <div className="mt-3 grid gap-2 text-xs text-gray-700 dark:text-gray-300">
            <div className="rounded-xl border border-gray-200 px-2.5 py-1.5 dark:border-gray-700">
              <span className="mr-2 font-medium text-gray-500 dark:text-gray-400">{t('graph.from')}</span>
              <span className="break-all">{clickedEdge.from}</span>
            </div>
            <div className="rounded-xl border border-gray-200 px-2.5 py-1.5 dark:border-gray-700">
              <span className="mr-2 font-medium text-gray-500 dark:text-gray-400">{t('graph.to')}</span>
              <span className="break-all">{clickedEdge.to}</span>
            </div>
          </div>
        ) : null}
      </div>
    );
  }

  // Footer layout
  if (clickedNode) {
    return (
      <div
        className={`flex items-start gap-3 rounded-lg px-3 py-2 text-left ${
          clickedNode.type === 'literal'
            ? 'border border-amber-200 bg-amber-50/60 dark:border-amber-900 dark:bg-amber-950/30'
            : 'border border-violet-200 bg-violet-50/60 dark:border-violet-900 dark:bg-violet-950/30'
        }`}
      >
        <div
          className="mt-1 h-3 w-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: clickedNode.type === 'literal' ? LITERAL_NODE_COLOR : RESOURCE_NODE_COLOR }}
        />
        <div className="min-w-0 flex-1">
          <div
            className="text-xs font-medium uppercase tracking-wide"
            style={{ color: clickedNode.type === 'literal' ? LITERAL_NODE_COLOR : RESOURCE_NODE_COLOR }}
          >
            {clickedNode.type}
          </div>
          <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-normal break-words">
            {clickedNode.label}
          </div>
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 break-all">
            {t('graph.idLabel')} {clickedNode.id}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-sm text-teal-600 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300 underline whitespace-nowrap flex-shrink-0"
        >
          {t('graph.clear')}
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 rounded-lg border border-teal-200 bg-teal-50/60 px-3 py-2 text-left dark:border-teal-900 dark:bg-teal-950/30">
      <div className="mt-1 h-3 w-3 rounded-full bg-teal-600 flex-shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="text-xs font-medium uppercase tracking-wide text-teal-700 dark:text-teal-300">
          {t('graph.edge')}
        </div>
        <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-normal break-words">
          {clickedEdge!.relation}
        </div>
        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 break-all">
          {t('graph.from')}: {clickedEdge!.from}
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400 break-all">
          {t('graph.to')}: {clickedEdge!.to}
        </div>
      </div>
      <button
        onClick={onClose}
        className="text-sm text-teal-600 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300 underline whitespace-nowrap flex-shrink-0"
      >
        {t('graph.clear')}
      </button>
    </div>
  );
}
