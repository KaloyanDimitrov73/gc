import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { D3Node, D3Link } from '../utils/graphGeometry';

export function useGraphSearch(
  d3NodesRef: React.RefObject<D3Node[]>,
  nodeSelectionRef: React.RefObject<d3.Selection<SVGGElement, D3Node, SVGGElement, unknown> | null>,
  linkSelectionRef: React.RefObject<d3.Selection<SVGLineElement, D3Link, SVGGElement, unknown> | null>,
  edgeLabelSelectionRef: React.RefObject<d3.Selection<SVGTextElement, D3Link, SVGGElement, unknown> | null>,
  svgRef: React.RefObject<SVGSVGElement | null>,
  zoomBehaviorRef: React.RefObject<d3.ZoomBehavior<SVGSVGElement, unknown> | null>,
) {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const matchingNodesRef = useRef<D3Node[]>([]);

  // Effect 1 — compute matches, update opacity, reset index
  useEffect(() => {
    const nodeSel = nodeSelectionRef.current;
    const linkSel = linkSelectionRef.current;
    const edgeLabelSel = edgeLabelSelectionRef.current;
    if (!nodeSel || !linkSel || !edgeLabelSel) return;

    const query = searchQuery.trim().toLowerCase();

    if (!query) {
      matchingNodesRef.current = [];
      setCurrentMatchIndex(0);
      nodeSel.style('opacity', '1');
      linkSel.style('opacity', '1');
      edgeLabelSel.style('opacity', '1');
      return;
    }

    const matchingNodes = d3NodesRef.current.filter((n: D3Node) =>
      n.label.toLowerCase().includes(query) || n.id.toLowerCase().includes(query)
    );
    matchingNodesRef.current = matchingNodes;
    setCurrentMatchIndex(0);

    const matchingIds = new Set(matchingNodes.map((n: D3Node) => n.id));

    nodeSel.style('opacity', d => (matchingIds.has(d.id) ? '1' : '0.15'));

    linkSel.style('opacity', d => {
      const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
      const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
      return matchingIds.has(src) || matchingIds.has(tgt) ? '1' : '0.08';
    });

    edgeLabelSel.style('opacity', d => {
      const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
      const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
      return matchingIds.has(src) || matchingIds.has(tgt) ? '1' : '0.08';
    });
  }, [searchQuery, d3NodesRef, nodeSelectionRef, linkSelectionRef, edgeLabelSelectionRef]);

  // Effect 2 — zoom to current match
  useEffect(() => {
    const node = matchingNodesRef.current[currentMatchIndex];
    const svgEl = svgRef.current;
    const zoom = zoomBehaviorRef.current;
    if (!node || node.x == null || node.y == null || !svgEl || !zoom) return;

    const svgSel = d3.select(svgEl);
    const width = svgEl.clientWidth;
    const height = svgEl.clientHeight;
    const scale = 1.2;
    const transform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(scale)
      .translate(-node.x, -node.y);
    svgSel.transition().duration(400).call(zoom.transform, transform);
  }, [currentMatchIndex, svgRef, zoomBehaviorRef]);

  const goToNext = () =>
    setCurrentMatchIndex((i: number) =>
      matchingNodesRef.current.length === 0 ? 0 : (i + 1) % matchingNodesRef.current.length
    );

  const goToPrev = () =>
    setCurrentMatchIndex((i: number) =>
      matchingNodesRef.current.length === 0
        ? 0
        : (i - 1 + matchingNodesRef.current.length) % matchingNodesRef.current.length
    );

  return {
    searchQuery,
    setSearchQuery,
    matchCount: matchingNodesRef.current.length,
    currentMatchIndex,
    goToNext,
    goToPrev,
  };
}
