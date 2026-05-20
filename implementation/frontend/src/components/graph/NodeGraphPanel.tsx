import { useState, useMemo, useEffect } from 'react';
import { useChat } from '../../contexts/ChatContext';
import { useSettings } from '../../contexts/SettingsContext';
import { useTheme } from '../../contexts/ThemeContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { mergeGraphNodes } from './utils/graphData';
import { DETAIL_PANEL_LAYOUT } from './utils/graphConstants';
import { useGraphViewport } from './hooks/useGraphViewport';
import { useGraphNodeExpansion } from './hooks/useGraphNodeExpansion';
import { useD3Graph } from './hooks/useD3Graph';
import { useGraphSearch } from './hooks/useGraphSearch';
import { GraphEmptyState } from './GraphEmptyState';
import { GraphToolbar } from './GraphToolbar';
import { GraphSearchBar } from './GraphSearchBar';
import { GraphDetailPanel } from './GraphDetailPanel';

export function NodeGraphPanel() {
  const { displayNodes: baseNodes, selectedMessageId } = useChat();
  const { retrievalMode } = useSettings();
  const { theme } = useTheme();
  const { t } = useLanguage();

  const [clickedNode, setClickedNode] = useState<{ id: string; label: string; type: 'resource' | 'literal' | 'document' | 'concept' } | null>(null);
  const [clickedEdge, setClickedEdge] = useState<{ from: string; to: string; relation: string } | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; id: string } | null>(null);

  // ── Expansion ──
  const {
    activeNeighborFragments,
    isNodeExpanding,
    toggleNodeExpansion,
    resetExpansion,
    hasExpandedNodes,
  } = useGraphNodeExpansion(setClickedNode, setClickedEdge);

  // ── Merged node graph ──
  const nodes = useMemo(
    () => mergeGraphNodes(baseNodes, Object.values(activeNeighborFragments)),
    [baseNodes, activeNeighborFragments],
  );

  // ── Reset everything on message change ──
  useEffect(() => {
    resetExpansion();
  }, [selectedMessageId, resetExpansion]);

  // ── Viewport ──
  const { graphViewport, handleGraphContainerRef } = useGraphViewport();

  // ── D3 rendering ──
  const {
    svgRef,
    zoomBehaviorRef,
    nodeSelectionRef,
    linkSelectionRef,
    edgeLabelSelectionRef,
    d3NodesRef,
    handleZoomIn,
    handleZoomOut,
    handleDownloadSvg,
  } = useD3Graph({
    nodes,
    selectedMessageId,
    theme,
    graphViewport,
    activeNeighborFragments,
    clickedNode,
    clickedEdge,
    setClickedNode,
    setClickedEdge,
    setTooltip,
    toggleNodeExpansion,
    t,
  });

  // ── Search ──
  const { searchQuery, setSearchQuery, matchCount, currentMatchIndex, goToNext, goToPrev } = useGraphSearch(
    d3NodesRef,
    nodeSelectionRef,
    linkSelectionRef,
    edgeLabelSelectionRef,
    svgRef,
    zoomBehaviorRef,
  );

  const showGraph = nodes.length > 0 && Boolean(selectedMessageId);

  return (
    <div className="min-h-0 min-w-0 flex-1 flex flex-col bg-gray-50 dark:bg-gray-950">

      {/* ── Graph area ── */}
      <div className="relative min-h-0 flex-1 overflow-hidden p-4 md:p-5 lg:p-6">
        {!showGraph ? (
          <GraphEmptyState />
        ) : (
          <div
            ref={handleGraphContainerRef}
            className="relative h-full min-h-[220px] overflow-hidden rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800 sm:min-h-[260px] lg:min-h-[320px]"
          >
            <GraphToolbar
              onDownload={handleDownloadSvg}
              onReset={resetExpansion}
              onZoomIn={handleZoomIn}
              onZoomOut={handleZoomOut}
              hasExpandedNodes={hasExpandedNodes}
            />

            {/* address rule 4.1.2 Name, Role, Value */}
            <svg
              ref={svgRef}
              className="w-full h-full"
              role="application"
              aria-label={`${t('graph.svgAriaPrefix')} ${nodes.length} ${t('graph.svgAriaSuffix')}`}
            />

            {/* Screen reader announcements */}
            <div aria-live="polite" aria-atomic="true" className="sr-only">
              {clickedNode
                ? `${t('graph.selectedNode')} ${clickedNode.type === 'literal' ? t('graph.literal') : t('graph.resource')} ${t('graph.nodeLabel')} ${clickedNode.label}. ${t('graph.idLabel')} ${clickedNode.id}`
                : clickedEdge
                  ? `${t('graph.selectedEdge')} ${clickedEdge.relation}, ${t('graph.fromLabel')} ${clickedEdge.from} ${t('graph.toLabel')} ${clickedEdge.to}`
                  : ''}
            </div>

            {/* Hover tooltip */}
            {/* address rule 1.4.13 Content on Hover or Focus */}
            {tooltip && (
              <div
                className="absolute pointer-events-none z-50 px-3 py-1.5 rounded-md bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-xs shadow-lg whitespace-nowrap"
                style={{ left: tooltip.x, top: tooltip.y, transform: 'translateX(-50%)' }}
              >
                {tooltip.id}
              </div>
            )}

            {DETAIL_PANEL_LAYOUT === 'overlay' && (
              <GraphDetailPanel
                clickedNode={clickedNode}
                clickedEdge={clickedEdge}
                isNodeExpanding={isNodeExpanding}
                onClose={() => { setClickedNode(null); setClickedEdge(null); }}
              />
            )}
          </div>
        )}
      </div>

      {/* ── Footer: search + status / detail panel ── */}
      <div className="relative z-10 shrink-0 flex flex-col gap-2 border-t border-gray-200 bg-white px-4 py-3 dark:border-gray-800 dark:bg-gray-900 md:px-5 lg:px-6">
        <GraphSearchBar
          value={searchQuery}
          onChange={setSearchQuery}
          onClear={() => setSearchQuery('')}
          matchCount={matchCount}
          currentMatchIndex={currentMatchIndex}
          onNext={goToNext}
          onPrev={goToPrev}
        />

        {DETAIL_PANEL_LAYOUT === 'footer' ? (
          <div className="min-w-0">
            <GraphDetailPanel
              clickedNode={clickedNode}
              clickedEdge={clickedEdge}
              isNodeExpanding={isNodeExpanding}
              onClose={() => { setClickedNode(null); setClickedEdge(null); }}
            />
            {!clickedNode && !clickedEdge && (
              <div className="text-sm text-gray-500 dark:text-gray-400 flex items-center gap-3 flex-wrap">
                <span>{t('graph.clickForDetails')}</span>
                <span className="text-gray-300 dark:text-gray-600">|</span>
                <span>{t('graph.nodes')}: {nodes.length}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-gray-500 dark:text-gray-400">
            <div className="flex flex-wrap items-center gap-3">
              <span>{t('graph.clickForDetails')}</span>
              <span className="text-gray-300 dark:text-gray-600">|</span>
              <span>{t('graph.nodes')}: {nodes.length}</span>
            </div>
            <span className="shrink-0">{retrievalMode === 'direct' ? t('graph.directMode') : t('graph.graphMode')}</span>
          </div>
        )}
      </div>
    </div>
  );
}
