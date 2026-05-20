import { useState, useCallback, useRef } from 'react';
import { apiClient } from '../../../services/api';
import type { Node } from '../../../types';

export type NodeSelection = { id: string; label: string; type: 'resource' | 'literal' | 'document' | 'concept' };

export function useGraphNodeExpansion(
  setClickedNode: (node: NodeSelection | null) => void,
  setClickedEdge: (edge: { from: string; to: string; relation: string } | null) => void,
) {
  const [activeNeighborFragments, setActiveNeighborFragments] = useState<Record<string, Node[]>>({});
  const [expandingNodeIds, setExpandingNodeIds] = useState<string[]>([]);
  const neighborCacheRef = useRef<Record<string, Node[]>>({});

  const isNodeExpanding = useCallback(
    (nodeId: string) => expandingNodeIds.includes(nodeId),
    [expandingNodeIds],
  );

  const toggleNodeExpansion = useCallback(
    async (node: NodeSelection) => {
      if (activeNeighborFragments[node.id]) {
        setActiveNeighborFragments((currentFragments) => {
          // Cascade collapse: remove this node's fragment and, recursively, the
          // fragments of nodes expanded *after* this one (deeper in expansion order).
          // Using Object.keys insertion order as a depth proxy prevents upward
          // propagation through bidirectional graph edges: a parent node's fragment
          // was added earlier (lower index) so it is never cascaded away.
          const expansionOrder = Object.keys(currentFragments);
          const toRemove = new Set<string>();
          const queue: Array<[string, number]> = [[node.id, -1]];
          while (queue.length > 0) {
            const [id, minIndex] = queue.shift()!;
            const idx = expansionOrder.indexOf(id);
            if (idx <= minIndex || toRemove.has(id)) continue;
            toRemove.add(id);
            const fragment = currentFragments[id];
            if (!fragment) continue;
            for (const n of fragment) {
              if (currentFragments[n.id] && !toRemove.has(n.id)) {
                queue.push([n.id, idx]);
              }
            }
          }
          const nextFragments = { ...currentFragments };
          for (const id of toRemove) {
            delete nextFragments[id];
          }
          return nextFragments;
        });
        return;
      }

      if (isNodeExpanding(node.id)) return;

      setExpandingNodeIds((currentIds) => [...currentIds, node.id]);
      try {
        const cachedFragment = neighborCacheRef.current[node.id];
        const fragmentNodes = cachedFragment ?? (await apiClient.getNodeNeighbors(node.id)).nodes;
        neighborCacheRef.current[node.id] = fragmentNodes;
        setActiveNeighborFragments((currentFragments) => ({
          ...currentFragments,
          [node.id]: fragmentNodes,
        }));
      } catch (error) {
        console.error(`Failed to expand node '${node.id}':`, error);
      } finally {
        setExpandingNodeIds((currentIds) => currentIds.filter((id) => id !== node.id));
      }
    },
    [activeNeighborFragments, isNodeExpanding],
  );

  const resetExpansion = useCallback(() => {
    setActiveNeighborFragments({});
    setExpandingNodeIds([]);
    setClickedEdge(null);
    setClickedNode(null);
    neighborCacheRef.current = {};
  }, [setClickedEdge, setClickedNode]);

  return {
    activeNeighborFragments,
    expandingNodeIds,
    isNodeExpanding,
    toggleNodeExpansion,
    resetExpansion,
    hasExpandedNodes: Object.keys(activeNeighborFragments).length > 0,
  };
}
