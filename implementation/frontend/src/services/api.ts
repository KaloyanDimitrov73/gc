/**
 * API client for communicating with the FastAPI backend.
 * Handles all HTTP requests and response transformations.
 */

const API_BASE_URL = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');

export interface QuestionRequest {
  question: string;
  retrievalMode: 'direct' | 'graph';
  llmModel: string;
  numberOfHubs: number;
  topicEntityId?: string;
  conversationId?: string;
  useDirectFinalAnswer?: boolean;
}

export interface NodeConnection {
  targetId: string;
  relation: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: 'resource' | 'literal';
  connections: NodeConnection[];
  x?: number;
  y?: number;
  score?: number;
}

export interface AnswerResponse {
  answer: string;
  nodes: GraphNode[];
  retrievalMode: string;
  sources: string[];
  messageId: string;
  guardrailsWarning?: string;
}

export type ProgressStep = 'input_validation' | 'retrieving' | 'preparing' | 'analyzing' | 'generating' | 'processing' | 'saving';

export interface ProgressStepEvent {
  type: 'progress';
  step: ProgressStep;
  percent: number;
  hubCompleted?: number | null;
  hubTotal?: number | null;
}

export interface CompleteStepEvent {
  type: 'complete';
  answer: string;
  nodes: GraphNode[];
  sources: string[];
  messageId: string;
  retrievalMode: string;
  guardrailsWarning?: string;
}

export interface ErrorStepEvent {
  type: 'error';
  code: string;
  detail: string;
}

export type StreamEvent = ProgressStepEvent | CompleteStepEvent | ErrorStepEvent;

export interface NodeNeighborsResponse {
  nodeId: string;
  nodes: GraphNode[];
}

export interface HealthResponse {
  status: string;
  hublinkAvailable: boolean;
  message: string;
  timestamp?: string;
}

export interface LlmModelInfo {
  model: string;
  provider: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface MessageRecord {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant';
  content: string;
  nodes?: GraphNode[];
  sources?: string[];
  createdAt: string;
}

export interface ConversationDetail extends Conversation {
  messages: MessageRecord[];
}

class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private buildHeaders(extraHeaders?: Record<string, string>): Record<string, string> {
    return {
      ...(extraHeaders || {}),
    };
  }

  private buildInit(extra?: RequestInit): RequestInit {
    return { credentials: 'include', ...extra };
  }

  /**
   * Ask a question and receive streaming SSE progress events.
   * Calls onProgress for each progress event. Returns the final AnswerResponse.
   */
  async askQuestionStreaming(
    request: QuestionRequest,
    onProgress: (event: ProgressStepEvent) => void,
  ): Promise<AnswerResponse> {
    const response = await fetch(`${this.baseUrl}/v1/qa/ask/stream`, this.buildInit({
      method: 'POST',
      headers: this.buildHeaders({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
      body: JSON.stringify(request),
    }));

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Streaming not supported by this browser or response.');
    }

    const decoder = new TextDecoder();
    const reader = response.body.getReader();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines
      const parts = buffer.split('\n\n');
      // Keep the last incomplete part in the buffer
      buffer = parts.pop() ?? '';

      for (const part of parts) {
        const dataLine = part.split('\n').find((l) => l.startsWith('data: '));
        if (!dataLine) continue;

        const json = dataLine.slice('data: '.length).trim();
        let event: StreamEvent;
        try {
          event = JSON.parse(json);
        } catch {
          continue;
        }

        if (event.type === 'progress') {
          onProgress(event);
        } else if (event.type === 'complete') {
          reader.cancel();
          return {
            answer: event.answer,
            nodes: event.nodes,
            retrievalMode: event.retrievalMode,
            sources: event.sources,
            messageId: event.messageId,
            guardrailsWarning: event.guardrailsWarning,
          };
        } else if (event.type === 'error') {
          reader.cancel();
          throw new Error(event.detail);
        }
      }
    }

    throw new Error('Stream ended without a complete event.');
  }

  /**
   * Ask a question and get an answer with knowledge graph.
   */
  async askQuestion(request: QuestionRequest): Promise<AnswerResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/qa/ask`, this.buildInit({
        method: 'POST',
        headers: this.buildHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(request),
      }));

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error asking question:', error);
      throw error;
    }
  }

  /**
   * Expand a graph node by fetching its direct neighbors from the backend.
   */
  async getNodeNeighbors(nodeId: string): Promise<NodeNeighborsResponse> {
    try {
      const params = new URLSearchParams({ nodeId });
      const response = await fetch(`${this.baseUrl}/v1/graph/node-neighbors?${params.toString()}`, this.buildInit({
        headers: this.buildHeaders(),
      }));

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching node neighbors:', error);
      throw error;
    }
  }

  /**
   * Check the health status of the backend service.
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/qa/health`, this.buildInit({
        headers: this.buildHeaders(),
      }));

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }

  /**
   * Fetch the list of available LLM models from the backend config.
   */
  async getLlmModels(): Promise<LlmModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/v1/qa/llm-models`, this.buildInit({
      headers: this.buildHeaders(),
    }));
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Trigger lazy initialization of the HubLink service.
   * Called once when the frontend first loads.
   */
  async initHublink(): Promise<{ status: string; hublinkAvailable: boolean }> {
    const response = await fetch(`${this.baseUrl}/v1/qa/init`, this.buildInit({
      method: 'POST',
      headers: this.buildHeaders(),
    }));

    if (!response.ok) {
      throw new Error(`Init failed: ${response.status}`);
    }

    const data = await response.json();
    return {
      status: data.status,
      hublinkAvailable: data.hublink_available,
    };
  }

  /**
   * List all chat conversations, newest first.
   */
  async listConversations(): Promise<Conversation[]> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/`, this.buildInit({
        headers: this.buildHeaders(),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error listing conversations:', error);
      throw error;
    }
  }

  /**
   * Create a new chat conversation. Title defaults to 'New Chat' if not provided.
   */
  async createConversation(title?: string): Promise<Conversation> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/`, this.buildInit({
        method: 'POST',
        headers: this.buildHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ title: title ?? null }),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error creating conversation:', error);
      throw error;
    }
  }

  /**
   * Update the title of an existing conversation.
   */
  async updateConversation(conversationId: string, title: string): Promise<Conversation> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/${conversationId}`, this.buildInit({
        method: 'PATCH',
        headers: this.buildHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ title }),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error updating conversation:', error);
      throw error;
    }
  }

  /**
   * Get a conversation with all its persisted messages.
   */
  async getConversation(conversationId: string): Promise<ConversationDetail> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/${conversationId}`, this.buildInit({
        headers: this.buildHeaders(),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching conversation:', error);
      throw error;
    }
  }

  /**
   * Delete a conversation and all its messages.
   */
  async deleteConversation(conversationId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/${conversationId}`, this.buildInit({
        method: 'DELETE',
        headers: this.buildHeaders(),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    } catch (error) {
      console.error('Error deleting conversation:', error);
      throw error;
    }
  }

  /**
   * Delete all conversations and their messages.
   */
  async deleteAllConversations(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/conversations/`, this.buildInit({
        method: 'DELETE',
        headers: this.buildHeaders(),
      }));
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    } catch (error) {
      console.error('Error deleting all conversations:', error);
      throw error;
    }
  }

  /**
   * Get basic service information.
   */
  async getServiceInfo(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/qa/health`, this.buildInit({
        headers: this.buildHeaders(),
      }));

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting service info:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export class for testing
export default APIClient;
