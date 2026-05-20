export type RetrievalMode = 'direct' | 'graph';

export type LLMModel = string;

export interface LlmModelInfo {
  model: string;
  provider: string;
}

export type NumberOfHubs = 10 | 20 | 30;

export interface NodeConnection {
  targetId: string;
  relation: string;
}

export interface Node {
  id: string;
  label: string;
  type: 'resource' | 'literal' | 'document' | 'concept';
  connections: NodeConnection[];
  x?: number;
  y?: number;
  score?: number;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  id: string;
  nodes?: Node[];
  isError?: boolean;
}
