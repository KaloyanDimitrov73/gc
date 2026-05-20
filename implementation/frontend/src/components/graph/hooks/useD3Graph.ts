import { useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import type { Node } from '../../../types';
import {
  NODE_RADIUS,
  LITERAL_NODE_WIDTH,
  LITERAL_NODE_HEIGHT,
  MAX_NODE_LABEL_LENGTH,
  MAX_EDGE_LABEL_LENGTH,
  EDGE_HIGHLIGHT_COLOR,
  RESOURCE_NODE_COLOR,
  LITERAL_NODE_COLOR,
  MIN_GRAPH_PADDING,
} from '../utils/graphConstants';
import {
  type D3Node,
  type D3Link,
  type GraphBounds,
  getNodeBoundaryPoint,
  getLinkMarker,
  clamp,
  clampNodeToBounds,
  getGraphBounds,
} from '../utils/graphGeometry';
import { truncateDisplayText } from '../utils/graphData';
import type { NodeSelection } from './useGraphNodeExpansion';

interface UseD3GraphParams {
  nodes: Node[];
  selectedMessageId: string | null;
  theme: 'light' | 'dark';
  graphViewport: { width: number; height: number };
  activeNeighborFragments: Record<string, Node[]>;
  clickedNode: NodeSelection | null;
  clickedEdge: { from: string; to: string; relation: string } | null;
  setClickedNode: (node: NodeSelection | null) => void;
  setClickedEdge: (edge: { from: string; to: string; relation: string } | null) => void;
  setTooltip: (tooltip: { x: number; y: number; id: string } | null) => void;
  toggleNodeExpansion: (node: NodeSelection) => void;
  t: (key: string) => string;
}

export function useD3Graph({
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
}: UseD3GraphParams) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<D3Node, D3Link> | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const nodeSelectionRef = useRef<d3.Selection<SVGGElement, D3Node, SVGGElement, unknown> | null>(null);
  const linkSelectionRef = useRef<d3.Selection<SVGLineElement, D3Link, SVGGElement, unknown> | null>(null);
  const edgeLabelSelectionRef = useRef<d3.Selection<SVGTextElement, D3Link, SVGGElement, unknown> | null>(null);
  const d3NodesRef = useRef<D3Node[]>([]);

  // Persists node positions across re-renders so existing nodes don't move
  const stablePositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  // Tracks the last message ID to detect when a genuinely new graph is loaded
  const previousMessageIdRef = useRef<string | null>(null);

  // Set to true whenever a new message is selected; cleared once the fit fires.
  // Using a ref (not local variable) survives re-renders before the simulation settles.
  const fitPendingRef = useRef(false);
  useEffect(() => {
    if (selectedMessageId !== null) {
      fitPendingRef.current = true;
    }
  }, [selectedMessageId]);

  // Mirrors of React state kept in refs so D3 event handlers never see stale closures
  const clickedEdgeRef = useRef<{ from: string; to: string; relation: string } | null>(null);
  const clickedNodeRef = useRef<NodeSelection | null>(null);

  useEffect(() => { clickedEdgeRef.current = clickedEdge; }, [clickedEdge]);
  useEffect(() => { clickedNodeRef.current = clickedNode; }, [clickedNode]);

  // ── Imperative highlight: node selection ──
  useEffect(() => {
    const nodeSel = nodeSelectionRef.current;
    if (!nodeSel) return;

    nodeSel.each(function (d) {
      const isActive = clickedNode?.id === d.id;
      const defaultStroke = d.type === 'literal' ? '#b45309' : '#6d28d9';
      d3.select(this)
        .selectAll<SVGCircleElement | SVGRectElement, D3Node>('circle, rect')
        .attr('stroke', isActive ? EDGE_HIGHLIGHT_COLOR : defaultStroke)
        .attr('stroke-width', isActive ? 6 : 3);
    });
  }, [clickedNode]);

  // ── Imperative highlight: edge selection ──
  useEffect(() => {
    const linkSel = linkSelectionRef.current;
    const edgeLabelSel = edgeLabelSelectionRef.current;
    if (!linkSel || !edgeLabelSel) return;

    const edgeColor = theme === 'dark' ? '#64748b' : '#94a3b8';

    linkSel
      .attr('stroke', d => {
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
        return clickedEdge?.from === src && clickedEdge?.to === tgt ? EDGE_HIGHLIGHT_COLOR : edgeColor;
      })
      .attr('stroke-width', d => {
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
        return clickedEdge?.from === src && clickedEdge?.to === tgt ? 4 : 2;
      })
      .attr('marker-end', d => {
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
        return getLinkMarker(d.targetType, Boolean(clickedEdge?.from === src && clickedEdge?.to === tgt));
      });

    edgeLabelSel.attr('fill', d => {
      const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
      const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
      if (clickedEdge?.from === src && clickedEdge?.to === tgt) return EDGE_HIGHLIGHT_COLOR;
      return theme === 'dark' ? '#cbd5e1' : '#486579';
    });
  }, [clickedEdge, theme]);

  // ── Zoom handlers ──
  const handleZoomIn = useCallback(() => {
    const svgEl = svgRef.current;
    const zoom = zoomBehaviorRef.current;
    if (!svgEl || !zoom) return;
    d3.select(svgEl).transition().duration(300).call(zoom.scaleBy, 1.4);
  }, []);

  const handleZoomOut = useCallback(() => {
    const svgEl = svgRef.current;
    const zoom = zoomBehaviorRef.current;
    if (!svgEl || !zoom) return;
    d3.select(svgEl).transition().duration(300).call(zoom.scaleBy, 1 / 1.4);
  }, []);

  // ── Download SVG ──
  const handleDownloadSvg = useCallback(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgEl);
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'retrieval-graph.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  // ── Main D3 rendering effect ──
  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl || nodes.length === 0) return;

    const width = graphViewport.width;
    const height = graphViewport.height;
    if (width <= 0 || height <= 0) return;

    // Detect whether this is a completely new graph (new message selected) or
    // just an expansion/collapse/theme-change. Only fit the view for new graphs.
    const isNewGraph = selectedMessageId !== null && selectedMessageId !== previousMessageIdRef.current;
    previousMessageIdRef.current = selectedMessageId;
    const dragPadding = MIN_GRAPH_PADDING;
    const nodeCount = nodes.length;
    const graphBounds = getGraphBounds(width, height, nodeCount, dragPadding);
    const graphWidth = graphBounds.maxX - graphBounds.minX;
    const graphHeight = graphBounds.maxY - graphBounds.minY;
    const graphScale = Math.max(
      graphWidth / Math.max(width - dragPadding * 2, 1),
      graphHeight / Math.max(height - dragPadding * 2, 1),
    );
    const linkDistance = Math.min(320, 180 * graphScale);
    const chargeStrength = -Math.min(1400, 400 * graphScale);

    // ── BFS depths & connected components ──
    const depthMap = (() => {
      const result = new Map<string, number>();
      const incomingCounts = new Map<string, number>();
      const undirectedAdj = new Map<string, string[]>();
      nodes.forEach(n => {
        if (!undirectedAdj.has(n.id)) undirectedAdj.set(n.id, []);
        if (!incomingCounts.has(n.id)) incomingCounts.set(n.id, 0);
        n.connections.forEach(c => {
          undirectedAdj.get(n.id)!.push(c.targetId);
          if (!undirectedAdj.has(c.targetId)) undirectedAdj.set(c.targetId, []);
          undirectedAdj.get(c.targetId)!.push(n.id);
          incomingCounts.set(c.targetId, (incomingCounts.get(c.targetId) ?? 0) + 1);
        });
      });
      const visited = new Set<string>();
      for (const n of nodes) {
        if (visited.has(n.id)) continue;
        const component: string[] = [];
        const queue = [n.id];
        visited.add(n.id);
        while (queue.length > 0) {
          const id = queue.shift()!;
          component.push(id);
          undirectedAdj.get(id)?.forEach(nb => { if (!visited.has(nb)) { visited.add(nb); queue.push(nb); } });
        }
        const roots = component.filter(id => (incomingCounts.get(id) ?? 0) === 0).sort();
        const bfsRoots = roots.length > 0 ? roots : [component.slice().sort()[0]];
        const depthQueue: string[] = [];
        bfsRoots.forEach(r => { result.set(r, 0); depthQueue.push(r); });
        while (depthQueue.length > 0) {
          const id = depthQueue.shift()!;
          const d = result.get(id)!;
          undirectedAdj.get(id)?.forEach(nb => { if (!result.has(nb)) { result.set(nb, d + 1); depthQueue.push(nb); } });
        }
      }
      return result;
    })();
    const depthValues = Array.from(depthMap.values()) as number[];
    const maxDepthValue = depthValues.reduce((a, b) => Math.max(a, b), 0);
    const canvasW = graphBounds.maxX - graphBounds.minX;
    const canvasH = graphBounds.maxY - graphBounds.minY;

    const componentMap = new Map<string, number>();
    {
      const adjacency = new Map<string, string[]>();
      nodes.forEach((n: Node) => {
        if (!adjacency.has(n.id)) adjacency.set(n.id, []);
        n.connections.forEach(c => {
          adjacency.get(n.id)!.push(c.targetId);
          if (!adjacency.has(c.targetId)) adjacency.set(c.targetId, []);
          adjacency.get(c.targetId)!.push(n.id);
        });
      });
      let compId = 0;
      for (const n of nodes) {
        if (componentMap.has(n.id)) continue;
        const queue = [n.id];
        while (queue.length > 0) {
          const id = queue.shift()!;
          if (componentMap.has(id)) continue;
          componentMap.set(id, compId);
          adjacency.get(id)?.forEach(nb => { if (!componentMap.has(nb)) queue.push(nb); });
        }
        compId++;
      }
    }

    const numComponents = (Array.from(componentMap.values()) as number[]).reduce((a, b) => Math.max(a, b), 0) + 1;
    const aspectRatio = canvasW / Math.max(canvasH, 1);
    const numCols = Math.max(1, Math.round(Math.sqrt(numComponents * aspectRatio)));
    const numRows = Math.ceil(numComponents / numCols);
    const compSlotWidth = canvasW / numCols;
    const compSlotHeight = canvasH / numRows;
    const componentCenterX = Array.from({ length: numComponents }, (_, i) =>
      graphBounds.minX + compSlotWidth * ((i % numCols) + 0.5),
    );
    const componentCenterY = Array.from({ length: numComponents }, (_, i) =>
      graphBounds.minY + compSlotHeight * (Math.floor(i / numCols) + 0.5),
    );

    const nodesByCompDepth = new Map<string, string[]>();
    nodes.forEach((n: Node) => {
      const key = `${componentMap.get(n.id) ?? 0}-${depthMap.get(n.id) ?? 0}`;
      if (!nodesByCompDepth.has(key)) nodesByCompDepth.set(key, []);
      nodesByCompDepth.get(key)!.push(n.id);
    });

    // ── Barycenter ordering to minimise edge crossings (Sugiyama-style) ──
    // For each layer, sort by average X-position of neighbours in the previous layer.
    // We run multiple passes (forward + backward) to iteratively reduce crossings.
    const layerOrderedX = new Map<string, number>(); // target X per node id
    {
      // Build directed adjacency (all edges, undirected here for placement)
      const adj = new Map<string, string[]>();
      nodes.forEach(n => {
        if (!adj.has(n.id)) adj.set(n.id, []);
        n.connections.forEach(c => {
          adj.get(n.id)!.push(c.targetId);
          if (!adj.has(c.targetId)) adj.set(c.targetId, []);
          adj.get(c.targetId)!.push(n.id);
        });
      });

      // Process each component independently
      for (let cId = 0; cId < numComponents; cId++) {
        const compLeft = graphBounds.minX + compSlotWidth * (cId % numCols);
        const compW = compSlotWidth;

        // Collect layers for this component
        const layers: string[][] = [];
        for (let d = 0; d <= maxDepthValue; d++) {
          const layer = nodesByCompDepth.get(`${cId}-${d}`) ?? [];
          layers.push([...layer].sort()); // deterministic initial order
        }

        // Current X positions within component [0..1]
        const posX = new Map<string, number>();
        layers.forEach(layer => {
          layer.forEach((id, i) => posX.set(id, (i + 0.5) / Math.max(layer.length, 1)));
        });

        // Barycenter passes: forward then backward, repeated
        const passes = 4;
        for (let pass = 0; pass < passes; pass++) {
          const forward = pass % 2 === 0;
          const layerRange = forward
            ? layers.map((_, i) => i).slice(1)
            : layers.map((_, i) => i).slice(0, -1).reverse();

          for (const layerIdx of layerRange) {
            const layer = layers[layerIdx];
            if (layer.length === 0) continue;

            // Compute barycenter for each node: avg X of its neighbours already placed
            const bary = layer.map(id => {
              const neighbours = adj.get(id) ?? [];
              const neighbourXs = neighbours
                .filter(nb => posX.has(nb))
                .map(nb => posX.get(nb)!);
              return neighbourXs.length > 0
                ? neighbourXs.reduce((s, x) => s + x, 0) / neighbourXs.length
                : posX.get(id) ?? 0.5;
            });

            // Sort layer by barycenter, keep relative order for ties (stable)
            const sorted = layer
              .map((id, i) => ({ id, b: bary[i] }))
              .sort((a, b) => a.b - b.b)
              .map(x => x.id);

            layers[layerIdx] = sorted;
            sorted.forEach((id, i) => posX.set(id, (i + 0.5) / Math.max(sorted.length, 1)));
          }
        }

        // Store final absolute X target positions
        layers.forEach(layer => {
          layer.forEach((id, i) => {
            const x = compLeft + compW * ((i + 0.5) / Math.max(layer.length, 1));
            layerOrderedX.set(id, x);
          });
        });
      }
    }

    // ── Build D3 data structures ──
    const d3Nodes: D3Node[] = nodes.map((n: Node) => {
      const stablePos = stablePositionsRef.current.get(n.id);
      const depth = depthMap.get(n.id) ?? 0;
      const compId = componentMap.get(n.id) ?? 0;
      const layerNodes = nodesByCompDepth.get(`${compId}-${depth}`) ?? [n.id];
      const posInLayer = layerNodes.indexOf(n.id);
      const slotLayerH = compSlotHeight / (maxDepthValue + 1);
      const initX = layerOrderedX.get(n.id) ?? (componentCenterX[compId] - compSlotWidth / 2 + (posInLayer + 0.5) * (compSlotWidth / layerNodes.length));
      const initY = componentCenterY[compId] - compSlotHeight / 2 + slotLayerH * (depth + 0.5);
      const node: D3Node = {
        id: n.id,
        label: n.label,
        displayLabel: truncateDisplayText(n.label, MAX_NODE_LABEL_LENGTH),
        type: n.type,
        x: stablePos?.x ?? initX,
        y: stablePos?.y ?? initY,
        // Fix existing nodes so they don't move; new nodes are free
        ...(stablePos ? { fx: stablePos.x, fy: stablePos.y } : {}),
      };
      clampNodeToBounds(node, graphBounds);
      return node;
    });

    const nodeById = new Map(d3Nodes.map(n => [n.id, n]));

    const d3Links: D3Link[] = [];
    nodes.forEach(node => {
      node.connections.forEach(conn => {
        if (nodeById.has(conn.targetId)) {
          d3Links.push({
            source: node.id,
            target: conn.targetId,
            relation: conn.relation,
            displayRelation: truncateDisplayText(conn.relation, MAX_EDGE_LABEL_LENGTH),
            targetType: nodeById.get(conn.targetId)?.type ?? 'resource',
          });
        }
      });
    });

    // ── Theme colours ──
    const isDark = theme === 'dark';
    const bgColor = isDark ? '#1f2937' : '#ffffff';
    const edgeColor = isDark ? '#64748b' : '#94a3b8';

    const getNodeColor = (type: string) =>
      type === 'literal' ? LITERAL_NODE_COLOR : RESOURCE_NODE_COLOR;

    const getNodeStroke = (type: string) =>
      type === 'literal' ? '#b45309' : '#6d28d9';

    // ── Clear previous render ──
    // Save current zoom transform so it can be restored after re-render
    const previousTransform = zoomBehaviorRef.current
      ? d3.zoomTransform(svgEl)
      : null;
    if (simulationRef.current) {
      simulationRef.current.stop();
      simulationRef.current = null;
    }
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.on('click', () => {
      setClickedEdge(null);
      setClickedNode(null);
    });
    svg.attr('width', width).attr('height', height).style('background', bgColor);

    // ── Arrow markers ──
    const defs = svg.append('defs');
    const addMarker = (id: string, color: string) => {
      defs.append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 9)
        .attr('refY', 0)
        .attr('markerWidth', 8)
        .attr('markerHeight', 8)
        .attr('markerUnits', 'strokeWidth')
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color);
    };
    addMarker('arrowhead-circle', edgeColor);
    addMarker('arrowhead-circle-active', EDGE_HIGHLIGHT_COLOR);
    addMarker('arrowhead-rect', edgeColor);
    addMarker('arrowhead-rect-active', EDGE_HIGHLIGHT_COLOR);

    // ── Zoom / pan container ──
    const g = svg.append('g');
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .extent([[0, 0], [width, height]])
      .on('zoom', (event) => { g.attr('transform', event.transform); });
    svg.call(zoomBehavior);
    // Restore previous zoom/pan so there is no jump after re-render.
    // Skip restoration for new graphs: fit-to-all-nodes will run after simulation ends.
    if (previousTransform && !isNewGraph) {
      svg.call(zoomBehavior.transform, previousTransform);
    }
    zoomBehaviorRef.current = zoomBehavior;

    // ── Draw links ──
    const linkGroup = g.append('g').attr('class', 'links');
    const linkSelection = linkGroup.selectAll<SVGLineElement, D3Link>('line')
      .data(d3Links)
      .enter()
      .append('line')
      .attr('stroke', edgeColor)
      .attr('stroke-width', 2)
      .attr('marker-end', d => getLinkMarker(d.targetType, false))
      .style('cursor', 'pointer')
      .on('mouseover', function (_, d) {
        d3.select(this)
          .attr('stroke', EDGE_HIGHLIGHT_COLOR)
          .attr('stroke-width', 4)
          .attr('marker-end', getLinkMarker(d.targetType, true));
      })
      .on('mouseout', function (_, d) {
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : d.source;
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : d.target;
        const isActive = clickedEdgeRef.current?.from === src && clickedEdgeRef.current?.to === tgt;
        d3.select(this)
          .attr('stroke', isActive ? EDGE_HIGHLIGHT_COLOR : edgeColor)
          .attr('stroke-width', isActive ? 4 : 2)
          .attr('marker-end', getLinkMarker(d.targetType, Boolean(isActive)));
      })
      .attr('tabindex', '0')
      .attr('role', 'button')
      .attr('aria-label', (d: D3Link) => `${t('graph.edgePrefix')}${d.relation}`)
      .on('click', function (event, d) {
        event.stopPropagation();
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
        const isActive = clickedEdgeRef.current?.from === src && clickedEdgeRef.current?.to === tgt;
        setClickedNode(null);
        setClickedEdge(isActive ? null : { from: src, to: tgt, relation: d.relation });
        if (isActive) (this as SVGLineElement).blur();
      })
      .on('keydown', (event, d) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          event.stopPropagation();
          const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
          const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
          const isActive = clickedEdgeRef.current?.from === src && clickedEdgeRef.current?.to === tgt;
          setClickedNode(null);
          setClickedEdge(isActive ? null : { from: src, to: tgt, relation: d.relation });
        } else if (event.key === 'Escape') {
          setClickedNode(null);
          setClickedEdge(null);
        }
      })
      .on('focus', function (_, d) {
        d3.select(this)
          .attr('stroke', EDGE_HIGHLIGHT_COLOR)
          .attr('stroke-width', 4)
          .attr('marker-end', getLinkMarker(d.targetType, true));
      })
      .on('blur', function (_, d) {
        const src = typeof d.source === 'object' ? (d.source as D3Node).id : (d.source as string);
        const tgt = typeof d.target === 'object' ? (d.target as D3Node).id : (d.target as string);
        const isActive = clickedEdgeRef.current?.from === src && clickedEdgeRef.current?.to === tgt;
        d3.select(this)
          .attr('stroke', isActive ? EDGE_HIGHLIGHT_COLOR : edgeColor)
          .attr('stroke-width', isActive ? 4 : 2)
          .attr('marker-end', getLinkMarker(d.targetType, Boolean(isActive)));
      });

    const edgeLabelSelection = linkGroup.selectAll<SVGTextElement, D3Link>('text')
      .data(d3Links)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', isDark ? '#cbd5e1' : '#486579')
      .attr('font-size', '10px')
      .attr('pointer-events', 'none')
      .text(d => d.displayRelation);

    // ── Draw nodes ──
    const nodeGroup = g.append('g').attr('class', 'nodes');
    const nodeSelection = nodeGroup.selectAll<SVGGElement, D3Node>('g')
      .data(d3Nodes)
      .enter()
      .append('g')
      .style('cursor', 'grab')
      .attr('tabindex', '0')
      .attr('role', 'button')
      .attr('aria-label', (d: D3Node) =>
        `${d.type === 'literal' ? t('graph.literal') : t('graph.resource')} ${t('graph.nodeLabel')}${d.label}`,
      )
      .on('click', (event, d) => {
        event.stopPropagation();
        setClickedEdge(null);
        setClickedNode(
          (prev: NodeSelection | null) =>
            prev?.id === d.id ? null : { id: d.id, label: d.label, type: d.type as NodeSelection['type'] },
        );
      })
      .on('dblclick', (event, d) => {
        event.stopPropagation();
        if (d.type === 'resource') {
          const isExpanded = Boolean(activeNeighborFragments[d.id]);
          setClickedNode(isExpanded ? null : { id: d.id, label: d.label, type: d.type });
          void toggleNodeExpansion({ id: d.id, label: d.label, type: d.type });
          if (isExpanded) (event.currentTarget as SVGGElement).blur();
        }
      })
      .on('keydown', (event, d) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          event.stopPropagation();
          setClickedEdge(null);
          setClickedNode(
            (prev: NodeSelection | null) =>
              prev?.id === d.id ? null : { id: d.id, label: d.label, type: d.type as NodeSelection['type'] },
          );
        } else if (event.key === 'e' || event.key === 'E') {
          if (d.type === 'resource') {
            event.preventDefault();
            event.stopPropagation();
            const isExpanded = Boolean(activeNeighborFragments[d.id]);
            setClickedNode(isExpanded ? null : { id: d.id, label: d.label, type: d.type });
            void toggleNodeExpansion({ id: d.id, label: d.label, type: d.type });
          }
        } else if (event.key === 'Escape') {
          setClickedNode(null);
          setClickedEdge(null);
        }
      })
      .on('focus', function () {
        d3.select(this)
          .selectAll<SVGCircleElement | SVGRectElement, D3Node>('circle, rect')
          .attr('stroke', EDGE_HIGHLIGHT_COLOR)
          .attr('stroke-width', 5);
      })
      .on('blur', function (_, d) {
        const isActive = clickedNodeRef.current?.id === d.id;
        d3.select(this)
          .selectAll<SVGCircleElement | SVGRectElement, D3Node>('circle, rect')
          .attr('stroke', getNodeStroke(d.type))
          .attr('stroke-width', isActive ? 6 : 3);
      });

    nodeSelectionRef.current = nodeSelection;
    linkSelectionRef.current = linkSelection;
    edgeLabelSelectionRef.current = edgeLabelSelection;
    d3NodesRef.current = d3Nodes;

    nodeSelection
      .filter(d => d.type !== 'literal')
      .append('circle')
      .attr('r', NODE_RADIUS)
      .attr('fill', d => getNodeColor(d.type))
      .attr('stroke', d => (clickedNodeRef.current?.id === d.id ? EDGE_HIGHLIGHT_COLOR : getNodeStroke(d.type)))
      .attr('stroke-width', d => (clickedNodeRef.current?.id === d.id ? 6 : 3))
      .on('mouseover', function (event, d) {
        d3.select(this).attr('stroke-width', 5);
        const svgRect = svgEl.getBoundingClientRect();
        setTooltip({ x: event.clientX - svgRect.left, y: event.clientY - svgRect.top - 50, id: d.id });
      })
      .on('mousemove', function (event, d) {
        const svgRect = svgEl.getBoundingClientRect();
        setTooltip({ x: event.clientX - svgRect.left, y: event.clientY - svgRect.top - 50, id: d.id });
      })
      .on('mouseout', function (_, d) {
        const isActive = clickedNodeRef.current?.id === d.id;
        d3.select(this).attr('stroke-width', isActive ? 6 : 3);
        setTooltip(null);
      });

    nodeSelection
      .filter(d => d.type === 'literal')
      .append('rect')
      .attr('x', -LITERAL_NODE_WIDTH / 2)
      .attr('y', -LITERAL_NODE_HEIGHT / 2)
      .attr('width', LITERAL_NODE_WIDTH)
      .attr('height', LITERAL_NODE_HEIGHT)
      .attr('rx', 12)
      .attr('ry', 12)
      .attr('fill', d => getNodeColor(d.type))
      .attr('stroke', d => (clickedNodeRef.current?.id === d.id ? EDGE_HIGHLIGHT_COLOR : getNodeStroke(d.type)))
      .attr('stroke-width', d => (clickedNodeRef.current?.id === d.id ? 6 : 3))
      .on('mouseover', function (event, d) {
        d3.select(this).attr('stroke-width', 5);
        const svgRect = svgEl.getBoundingClientRect();
        setTooltip({ x: event.clientX - svgRect.left, y: event.clientY - svgRect.top - 50, id: d.id });
      })
      .on('mousemove', function (event, d) {
        const svgRect = svgEl.getBoundingClientRect();
        setTooltip({ x: event.clientX - svgRect.left, y: event.clientY - svgRect.top - 50, id: d.id });
      })
      .on('mouseout', function (_, d) {
        const isActive = clickedNodeRef.current?.id === d.id;
        d3.select(this).attr('stroke-width', isActive ? 6 : 3);
        setTooltip(null);
      });

    // ── Labels (word-wrapped) ──
    nodeSelection.each(function (d) {
      const el = d3.select(this);
      const words = d.displayLabel.split(' ');
      const lines: string[] = [];
      let currentLine = '';
      const maxLineLength = d.type === 'literal' ? 14 : 10;

      words.forEach(word => {
        const testLine = currentLine ? currentLine + ' ' + word : word;
        if (testLine.length > maxLineLength) {
          if (currentLine) lines.push(currentLine);
          currentLine = word;
        } else {
          currentLine = testLine;
        }
      });
      if (currentLine) lines.push(currentLine);

      const lineHeight = 13;
      const startY = -((lines.length - 1) * lineHeight) / 2;

      lines.forEach((line, i) => {
        el.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', startY + i * lineHeight)
          .attr('fill', '#ffffff')
          .attr('font-size', '11px')
          .attr('pointer-events', 'none')
          .text(line);
      });
    });

    // ── Force simulation ──
    // fitPendingRef is set when selectedMessageId changes and cleared once the fit fires.

    const simulation = d3.forceSimulation<D3Node>(d3Nodes)
      .force('link', d3.forceLink<D3Node, D3Link>(d3Links).id(d => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('collision', d3.forceCollide<D3Node>().radius(d => d.type === 'literal' ? 72 : 50))
      .on('tick', () => {
        d3Nodes.forEach(node => clampNodeToBounds(node, graphBounds));

        linkSelection
          .attr('x1', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            return getNodeBoundaryPoint(source, target.x ?? 0, target.y ?? 0).x;
          })
          .attr('y1', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            return getNodeBoundaryPoint(source, target.x ?? 0, target.y ?? 0).y;
          })
          .attr('x2', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            return getNodeBoundaryPoint(target, source.x ?? 0, source.y ?? 0).x;
          })
          .attr('y2', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            return getNodeBoundaryPoint(target, source.x ?? 0, source.y ?? 0).y;
          });

        edgeLabelSelection
          .attr('x', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            const start = getNodeBoundaryPoint(source, target.x ?? 0, target.y ?? 0);
            const end = getNodeBoundaryPoint(target, source.x ?? 0, source.y ?? 0);
            return (start.x + end.x) / 2;
          })
          .attr('y', d => {
            const source = d.source as D3Node;
            const target = d.target as D3Node;
            const start = getNodeBoundaryPoint(source, target.x ?? 0, target.y ?? 0);
            const end = getNodeBoundaryPoint(target, source.x ?? 0, source.y ?? 0);
            return (start.y + end.y) / 2;
          });

        nodeSelection.attr('transform', d => `translate(${d.x},${d.y})`);

        // Early fit: once the graph is mostly settled, zoom to fit all nodes.
        // fitPendingRef survives re-renders; it is set when selectedMessageId changes
        // and cleared here so the fit fires exactly once per new graph.
        if (fitPendingRef.current && simulation.alpha() < 0.05 && zoomBehaviorRef.current) {
          fitPendingRef.current = false;
          const xs = d3Nodes.map(n => n.x ?? 0);
          const ys = d3Nodes.map(n => n.y ?? 0);
          const bboxMinX = Math.min(...xs);
          const bboxMaxX = Math.max(...xs);
          const bboxMinY = Math.min(...ys);
          const bboxMaxY = Math.max(...ys);
          const bboxW = bboxMaxX - bboxMinX;
          const bboxH = bboxMaxY - bboxMinY;
          const fitPadding = NODE_RADIUS * 2 + 20;
          const scaleX = bboxW > 0 ? (width - fitPadding * 2) / bboxW : 1;
          const scaleY = bboxH > 0 ? (height - fitPadding * 2) / bboxH : 1;
          const clampedScale = Math.min(Math.max(Math.min(scaleX, scaleY), 0.2), 4);
          const midX = (bboxMinX + bboxMaxX) / 2;
          const midY = (bboxMinY + bboxMaxY) / 2;
          const fitTransform = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(clampedScale)
            .translate(-midX, -midY);
          d3.select(svgEl)
            .transition()
            .duration(400)
            .call(zoomBehaviorRef.current.transform, fitTransform);
        }
      });

    simulationRef.current = simulation;

    // When simulation settles: fix all nodes in place and persist positions
    simulation.on('end', () => {
      d3Nodes.forEach(node => {
        const x = node.x ?? 0;
        const y = node.y ?? 0;
        node.fx = x;
        node.fy = y;
        stablePositionsRef.current.set(node.id, { x, y });
      });
    });

    for (let cId = 0; cId < numComponents; cId++) {
      // Strong Y force: keep nodes in their depth layer
      simulation.force(
        `layer-y-${cId}`,
        d3.forceY<D3Node>(d => {
          const depth = depthMap.get(d.id) ?? 0;
          const slotLayerH = compSlotHeight / (maxDepthValue + 1);
          return componentCenterY[cId] - compSlotHeight / 2 + slotLayerH * (depth + 0.5);
        }).strength(d => ((componentMap.get(d.id) ?? 0) === cId ? 0.9 : 0)),
      );
      // Medium X force: pull toward barycenter-ordered position
      simulation.force(
        `layer-x-${cId}`,
        d3.forceX<D3Node>(d => layerOrderedX.get(d.id) ?? componentCenterX[cId])
          .strength(d => ((componentMap.get(d.id) ?? 0) === cId ? 0.4 : 0)),
      );
    }

    // ── Drag behaviour ──
    const drag = d3.drag<SVGGElement, D3Node>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = clamp(event.x, graphBounds.minX, graphBounds.maxX);
        d.fy = clamp(event.y, graphBounds.minY, graphBounds.maxY);
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        const x = clamp(event.x, graphBounds.minX, graphBounds.maxX);
        const y = clamp(event.y, graphBounds.minY, graphBounds.maxY);
        d.fx = x;
        d.fy = y;
        stablePositionsRef.current.set(d.id, { x, y });
      });

    nodeSelection.call(drag);

    return () => { simulation.stop(); };
  }, [nodes, theme, graphViewport, t, toggleNodeExpansion, activeNeighborFragments, setClickedNode, setClickedEdge, setTooltip]);

  return {
    svgRef,
    zoomBehaviorRef,
    nodeSelectionRef,
    linkSelectionRef,
    edgeLabelSelectionRef,
    d3NodesRef,
    handleZoomIn,
    handleZoomOut,
    handleDownloadSvg,
  };
}
