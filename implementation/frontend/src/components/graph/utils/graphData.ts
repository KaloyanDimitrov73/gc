import type { Node } from '../../../types';

export function truncateDisplayText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  const truncated = text.slice(0, maxLength - 3).trimEnd();
  return `${truncated}...`;
}

export function getConnectionKey(connection: Node['connections'][number]): string {
  return `${connection.targetId}::${connection.relation}`;
}

export function mergeGraphNodes(baseNodes: Node[], fragments: Node[][]): Node[] {
  const nodesById = new Map<string, Node>();

  const mergeNode = (node: Node) => {
    const existing = nodesById.get(node.id);
    if (!existing) {
      nodesById.set(node.id, { ...node, connections: [...node.connections] });
      return;
    }

    if (existing.label === existing.id && node.label) existing.label = node.label;
    if (existing.x == null && node.x != null) existing.x = node.x;
    if (existing.y == null && node.y != null) existing.y = node.y;
    if (existing.score == null && node.score != null) existing.score = node.score;

    const existingConnectionKeys = new Set(existing.connections.map(getConnectionKey));
    node.connections.forEach((connection) => {
      const connectionKey = getConnectionKey(connection);
      if (existingConnectionKeys.has(connectionKey)) return;
      existing.connections.push({ ...connection });
      existingConnectionKeys.add(connectionKey);
    });
  };

  baseNodes.forEach(mergeNode);
  fragments.forEach((fragment) => fragment.forEach(mergeNode));
  return Array.from(nodesById.values());
}
