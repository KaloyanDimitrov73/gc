import type * as d3 from 'd3';
import { NODE_RADIUS, LITERAL_NODE_WIDTH, LITERAL_NODE_HEIGHT, GRAPH_GROWTH_PER_NODE } from './graphConstants';

export interface GraphBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface D3Node extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  displayLabel: string;
  type: 'resource' | 'literal'  | 'document' | 'concept';
}

export interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  relation: string;
  displayRelation: string;
  targetType: 'resource' | 'literal'  | 'document' | 'concept';
}

export function getNodeBoundaryPoint(node: D3Node, towardX: number, towardY: number) {
  const x = node.x ?? 0;
  const y = node.y ?? 0;
  const dx = towardX - x;
  const dy = towardY - y;
  const distance = Math.hypot(dx, dy) || 1;

  if (node.type !== 'literal') {
    const scale = NODE_RADIUS / distance;
    return { x: x + dx * scale, y: y + dy * scale };
  }

  const halfWidth = LITERAL_NODE_WIDTH / 2;
  const halfHeight = LITERAL_NODE_HEIGHT / 2;
  const scale = 1 / Math.max(Math.abs(dx) / halfWidth || 0, Math.abs(dy) / halfHeight || 0, Number.EPSILON);

  return { x: x + dx * scale, y: y + dy * scale };
}

export function getLinkMarker(targetType: 'resource' | 'literal'  | 'document' | 'concept', isActive: boolean): string {
  if (targetType === 'literal') {
    return isActive ? 'url(#arrowhead-rect-active)' : 'url(#arrowhead-rect)';
  }
  return isActive ? 'url(#arrowhead-circle-active)' : 'url(#arrowhead-circle)';
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function clampNodeToBounds(node: D3Node, bounds: GraphBounds) {
  node.x = clamp(node.x ?? bounds.minX, bounds.minX, bounds.maxX);
  node.y = clamp(node.y ?? bounds.minY, bounds.minY, bounds.maxY);

  if (node.fx != null) node.fx = clamp(node.fx, bounds.minX, bounds.maxX);
  if (node.fy != null) node.fy = clamp(node.fy, bounds.minY, bounds.maxY);
}

export function getGraphBounds(
  width: number,
  height: number,
  nodeCount: number,
  minPadding: number,
): GraphBounds {
  const minGraphWidth = Math.max(width - minPadding * 2, NODE_RADIUS * 4);
  const minGraphHeight = Math.max(height - minPadding * 2, NODE_RADIUS * 4);
  const graphWidth = minGraphWidth + nodeCount * GRAPH_GROWTH_PER_NODE;
  const graphHeight = minGraphHeight + nodeCount * GRAPH_GROWTH_PER_NODE;
  const centerX = width / 2;
  const centerY = height / 2;

  return {
    minX: centerX - graphWidth / 2,
    maxX: centerX + graphWidth / 2,
    minY: centerY - graphHeight / 2,
    maxY: centerY + graphHeight / 2,
  };
}
