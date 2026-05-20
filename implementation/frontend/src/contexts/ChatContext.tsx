import { createContext, useContext, useState, useEffect, useCallback, ReactNode, useMemo } from 'react';
import { apiClient } from '../services/api';
import { useSettings } from './SettingsContext';
import type { Message, Node } from '../types';
import type { GraphNode, Conversation, MessageRecord, ProgressStepEvent, ProgressStep } from '../services/api';

// Per-conversation state bucket. Each conversation owns its own messages,
// loading flag, and selected message — independent of what is currently displayed.
interface ConversationState {
  messages: Message[];
  isLoading: boolean;
  selectedMessageId: string | null;
  progress: number | null;
  currentStep: ProgressStep | null;
  hubCompleted: number | null;
  hubTotal: number | null;
}

const EMPTY_CONV: ConversationState = { messages: [], isLoading: false, selectedMessageId: null, progress: null, currentStep: null, hubCompleted: null, hubTotal: null };

interface ChatContextValue {
  messages: Message[];
  selectedMessageId: string | null;
  isLoading: boolean;
  progress: number | null;
  currentStep: ProgressStep | null;
  hubCompleted: number | null;
  hubTotal: number | null;
  displayNodes: Node[];
  sendMessage: (content: string, topicEntityId?: string) => Promise<void>;
  clearChat: () => Promise<void>;
  selectMessage: (messageId: string) => void;
  // Conversation management
  currentConversationId: string | null;
  conversations: Conversation[];
  isConversationsLoading: boolean;
  isConversationLoading: (id: string) => boolean;
  createConversation: () => Promise<void>;
  switchConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

function mapApiNodeToUiNode(node: GraphNode): Node {
  return {
    id: node.id,
    label: node.label,
    type: node.type,
    connections: node.connections.map((conn) => ({
      targetId: conn.targetId,
      relation: conn.relation,
    })),
    x: node.x,
    y: node.y,
    score: node.score,
  };
}

function mapMessageRecordToMessage(record: MessageRecord): Message {
  return {
    id: record.id,
    role: record.role,
    content: record.content,
    nodes: record.nodes ? record.nodes.map(mapApiNodeToUiNode) : undefined,
  };
}

const WELCOME_MESSAGE: Message = {
  id: 'msg-welcome',
  role: 'assistant',
  content: 'Hello! I am your chat assistant. Please type your question to start the conversation (see the information above for guidance).',
};

const INITIAL_MESSAGES: Message[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: 'What is the definition of the Client-Server software architecture pattern?',
  },
  {
    id: 'msg-2',
    role: 'assistant',
    content:
      'The Client-Server architecture pattern is a distributed application structure that partitions tasks or workloads between providers of a resource or service (servers) and service requesters (clients) [1,2].\n\nKey characteristics [2]:\n\u2022 Clients initiate requests for services or resources\n\u2022 Servers respond to these requests and provide the requested services\n\u2022 Communication occurs over a network using defined protocols\n\u2022 Clear separation of concerns between presentation (client) and data management (server)\n\u2022 Supports multiple clients connecting to a single server simultaneously\n\nThis architectural pattern is widely used in web applications, database systems, email services, and many other networked applications [1].\n\n [1] Source: Design Patterns in Software Architecture, DOI: 10.1109/TSE.2018.1234567 \n [2] Source: Distributed Systems Fundamentals, DOI: 10.1145/3359591.3359592 \n\n PLEASE NOTE: The DOIs in the generated answer are not real and are just provided for demonstration purposes only. \n Click on "View Graph" to explore the graph.',
    nodes: [
      {
        id: 'doc-1',
        label: 'Software Architecture Patterns',
        type: 'document',
        connections: [{ targetId: 'concept-1', relation: 'defines' }],
      },
      {
        id: 'concept-1',
        label: 'Client-Server',
        type: 'concept',
        connections: [
          { targetId: 'doc-2', relation: 'requires' },
          { targetId: 'doc-3', relation: 'uses' },
        ],
      },
      {
        id: 'doc-2',
        label: 'Distributed Systems',
        type: 'document',
        connections: [],
      },
      {
        id: 'doc-3',
        label: 'Network Protocols',
        type: 'document',
        connections: [],
      },
      {
        id: 'doc-4',
        label: 'Design Patterns Book',
        type: 'document',
        connections: [{ targetId: 'concept-1', relation: 'describes' }],
      },
    ],
  },
];



export function ChatProvider({ children }: { children: ReactNode }) {
  const { retrievalMode, llmModel, numberOfHubs, useDirectFinalAnswer } = useSettings();

  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isConversationsLoading, setIsConversationsLoading] = useState(false);

  // Per-conversation state: keyed by conversation ID.
  // In-flight requests update their own bucket regardless of which conversation is displayed.
  const [convStates, setConvStates] = useState<Record<string, ConversationState>>({});

  // Track selected message for initial (demo) messages when no conversation is active
  const [initialSelectedMessageId, setInitialSelectedMessageId] = useState<string | null>('msg-2');

  // Derived: what the current view sees
  const currentConv = convStates[currentConversationId ?? ''] ?? EMPTY_CONV;
  const messages = currentConversationId ? currentConv.messages : INITIAL_MESSAGES;
  const isLoading = currentConv.isLoading;
  const progress = currentConv.progress;
  const currentStep = currentConv.currentStep;
  const hubCompleted = currentConv.hubCompleted;
  const hubTotal = currentConv.hubTotal;
  const selectedMessageId = currentConversationId
    ? currentConv.selectedMessageId
    : initialSelectedMessageId;

  // Immutable helper: update one conversation's state without touching others
  const updateConv = useCallback((id: string, updater: (prev: ConversationState) => ConversationState) => {
    setConvStates(prev => ({
      ...prev,
      [id]: updater(prev[id] ?? EMPTY_CONV),
    }));
  }, []);

  // Load conversation list on mount
  useEffect(() => {
    setIsConversationsLoading(true);
    apiClient
      .listConversations()
      .then((data) => setConversations(data))
      .catch((err) => console.error('Failed to load conversations:', err))
      .finally(() => setIsConversationsLoading(false));
  }, []);

  const selectedMessage = useMemo(
    () => messages.find((m) => m.id === selectedMessageId),
    [messages, selectedMessageId]
  );

  const displayNodes = useMemo(() => selectedMessage?.nodes || [], [selectedMessage]);

  const sendMessage = async (content: string, topicEntityId?: string) => {
    const userMsg: Message = { id: `msg-${Date.now()}`, role: 'user', content };

    // Capture the conversation this send belongs to. This closure variable never changes
    // for this invocation, so the request always updates the correct conversation's state
    // even if the user navigates away mid-flight.
    let conversationId = currentConversationId;

    // Track whether this is the first message in an existing conversation so we can
    // rename it from the default "New Chat" title.
    const isFirstMessageInExisting =
      conversationId !== null &&
      !(convStates[conversationId]?.messages ?? []).some((m: Message) => m.role === 'user');

    if (conversationId) {
      updateConv(conversationId, c => ({ ...c, messages: [...c.messages, userMsg], isLoading: true, progress: null, currentStep: null }));
    }

    if (!conversationId) {
      try {
        const title = content.slice(0, 50);
        const newConversation = await apiClient.createConversation(title);
        conversationId = newConversation.id;
        setConversations(prev => [newConversation, ...prev]);
        // Only make the auto-created conversation current if the user hasn't already
        // navigated somewhere else (i.e., currentConversationId is still null)
        setCurrentConversationId(prev => prev ?? conversationId);
        updateConv(conversationId, () => ({ messages: [userMsg], isLoading: true, selectedMessageId: null, progress: null, currentStep: null, hubCompleted: null, hubTotal: null }));
      } catch (err) {
        console.error('Failed to create conversation — continuing without persistence:', err);
        // No conversationId: proceed without persistence, no state to update
        return;
      }
    }

    // If this is the first message sent to a "New Chat" conversation created via the
    // sidebar button, rename it using the message content.
    if (isFirstMessageInExisting && conversationId) {
      const newTitle = content.slice(0, 50);
      apiClient.updateConversation(conversationId, newTitle)
        .then(updated => setConversations(prev => prev.map(c => c.id === updated.id ? updated : c)))
        .catch(() => {/* non-fatal: title stays as "New Chat" */});
    }

    try {
      const response = await apiClient.askQuestionStreaming(
        {
          question: content,
          retrievalMode,
          llmModel,
          numberOfHubs,
          topicEntityId: topicEntityId?.trim() ? topicEntityId.trim() : undefined,
          conversationId: conversationId ?? undefined,
          useDirectFinalAnswer,
        },
        (event) => {
          updateConv(conversationId!, c => {
            const stepChanged = event.step !== c.currentStep;
            return {
              ...c,
              progress: event.percent,
              currentStep: event.step,
              hubCompleted: event.hubCompleted != null ? event.hubCompleted : (stepChanged ? null : c.hubCompleted),
              hubTotal: event.hubTotal != null ? event.hubTotal : (stepChanged ? null : c.hubTotal),
            };
          });
        },
      );

      const mappedNodes = response.nodes.map(mapApiNodeToUiNode);
      const assistantMsg: Message = {
        id: response.messageId,
        role: 'assistant',
        content: response.answer,
        nodes: mappedNodes,
      };

      updateConv(conversationId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        isLoading: false,
        progress: null,
        currentStep: null,
        hubCompleted: null,
        hubTotal: null,
        selectedMessageId: assistantMsg.id,
      }));

      // Refresh sidebar ordering (updated_at changed)
      apiClient.listConversations()
        .then(data => setConversations(data))
        .catch(() => {/* non-fatal */});
    } catch (error) {
      console.error('Error getting answer:', error);

      const errorMsg: Message = {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error while processing your question: ${error instanceof Error ? error.message : 'Unknown error'}`,
        nodes: [],
        isError: true,
      };
      updateConv(conversationId, c => ({
        ...c,
        messages: [...c.messages, errorMsg],
        isLoading: false,
        progress: null,
        currentStep: null,
        hubCompleted: null,
        hubTotal: null,
      }));
    }
  };

  /**
   * Clear all chats and delete every conversation from the backend.
   */
  const clearChat = async () => {
    try {
      await apiClient.deleteAllConversations();
    } catch (err) {
      console.error('Failed to delete all conversations on clear:', err);
    }
    setConvStates({});
    setConversations([]);
    setCurrentConversationId(null);
  };

  const selectMessage = (messageId: string) => {
    if (!currentConversationId) {
      setInitialSelectedMessageId(messageId);
      return;
    }
    updateConv(currentConversationId, c => ({ ...c, selectedMessageId: messageId }));
  };

  /**
   * Create a fresh conversation and switch to it.
   * Any in-flight request in the previous conversation continues unaffected.
   */
  const createConversation = async () => {
    try {
      const newConversation = await apiClient.createConversation();
      setConversations(prev => [newConversation, ...prev]);
      updateConv(newConversation.id, () => ({ ...EMPTY_CONV, messages: [WELCOME_MESSAGE] }));
      setCurrentConversationId(newConversation.id);
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  };

  /**
   * Switch to an existing conversation.
   * If its messages are already cached in convStates, the switch is instant.
   * In-flight requests in other conversations continue unaffected.
   */
  const switchConversation = async (conversationId: string) => {
    setCurrentConversationId(conversationId);

    // Already cached from this session — show instantly
    if (convStates[conversationId]?.messages.length > 0) return;

    try {
      const detail = await apiClient.getConversation(conversationId);
      updateConv(conversationId, () => ({
        messages: detail.messages.map(mapMessageRecordToMessage),
        isLoading: false,
        selectedMessageId: null,
      }));
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  };

  /**
   * Delete a conversation. If it was active, clear the current view.
   * In-flight requests for the deleted conversation will complete but their
   * state updates become orphaned (no-op for the UI).
   */
  const deleteConversation = async (conversationId: string) => {
    try {
      await apiClient.deleteConversation(conversationId);
      setConversations(prev => prev.filter(c => c.id !== conversationId));
      setConvStates(prev => {
        const next = { ...prev };
        delete next[conversationId];
        return next;
      });
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const isConversationLoading = useCallback(
    (id: string) => convStates[id]?.isLoading ?? false,
    [convStates]
  );

  return (
    <ChatContext.Provider
      value={{
        messages,
        selectedMessageId,
        isLoading,
        progress,
        currentStep,
        hubCompleted,
        hubTotal,
        displayNodes,
        sendMessage,
        clearChat,
        selectMessage,
        currentConversationId,
        conversations,
        isConversationsLoading,
        isConversationLoading,
        createConversation,
        switchConversation,
        deleteConversation,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatContextValue {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
