import { useEffect, useState } from 'react';
import {
  MessageSquare,
  Trash2,
  Plus,
  Circle,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  X,
} from 'lucide-react';
import { useChat } from '../contexts/ChatContext';
import { useHealth } from '../contexts/HealthContext';
import { useLanguage } from '../contexts/LanguageContext';

interface SidebarProps {
  isDrawerMode: boolean;
  onRequestClose?: () => void;
}

export function Sidebar({ isDrawerMode, onRequestClose }: SidebarProps) {
  const {
    clearChat,
    conversations,
    isConversationsLoading,
    currentConversationId,
    isConversationLoading,
    createConversation,
    switchConversation,
    deleteConversation,
  } = useChat();
  const { backendStatus } = useHealth();
  const { t } = useLanguage();
  const [collapsed, setCollapsed] = useState(false);
  const [sidebarSearch, setSidebarSearch] = useState('');

  useEffect(() => {
    if (isDrawerMode) {
      setCollapsed(false);
    }
  }, [isDrawerMode]);

  const isCollapsed = !isDrawerMode && collapsed;

  // address rule 1.4.1 Use of Color
  const getStatusColor = () => {
    switch (backendStatus) {
      case 'hublink':
        return 'text-green-400';
      case 'mock':
        return 'text-yellow-400';
      case 'offline':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusText = () => {
    switch (backendStatus) {
      case 'hublink':
        return t('sidebar.connected');
      case 'mock':
        return t('sidebar.mockMode');
      case 'offline':
        return t('sidebar.backendOffline');
      case 'initializing':
        return t('sidebar.backendInitializing');
      default:
        return t('sidebar.checking');
    }
  };

  const filteredConversations = conversations.filter(
    (c) => !sidebarSearch || c.title.toLowerCase().includes(sidebarSearch.toLowerCase()),
  );

  const closeDrawerIfNeeded = () => {
    if (isDrawerMode) {
      onRequestClose?.();
    }
  };

  const handleNewChat = async () => {
    await createConversation();
    closeDrawerIfNeeded();
  };

  const handleSwitchConversation = async (conversationId: string) => {
    await switchConversation(conversationId);
    closeDrawerIfNeeded();
  };

  const handleDeleteConversation = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    await deleteConversation(conversationId);
  };

  return (
    <div
      className={`dark flex h-full flex-col border-r border-gray-800 bg-gray-900 text-white transition-all duration-300 dark:border-gray-800 dark:bg-gray-950 ${
        isCollapsed ? 'w-14' : isDrawerMode ? 'w-72 max-w-[85vw]' : 'w-56'
      }`}
    >
      <div className="flex items-center justify-end border-b border-gray-800 p-4 dark:border-gray-800">
        {/* address rule 1.4.11 Non-text Contrast */}
        {/* address rule 2.5.2 Pointer Cancellation */}
        {/* address rule 4.1.2 Name, Role, Value */}
        {isDrawerMode ? (
          <button
            onClick={closeDrawerIfNeeded}
            aria-label={t('sidebar.closeSidebar')}
            className="group relative rounded-lg p-1.5 transition-colors hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
          >
            <X className="h-4 w-4 text-gray-400" />
            {/* address rule 1.4.13 Content on Hover or Focus */}
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
              {t('sidebar.closeSidebar')}
            </span>
          </button>
        ) : (
          <button
            onClick={() => setCollapsed((value) => !value)}
            aria-label={isCollapsed ? t('sidebar.expandSidebar') : t('sidebar.collapseSidebar')}
            aria-expanded={!isCollapsed}
            className="group relative rounded-lg p-1.5 transition-colors hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
          >
            {isCollapsed ? (
              <PanelLeftOpen className="h-4 w-4 text-gray-400" />
            ) : (
              <PanelLeftClose className="h-4 w-4 text-gray-400" />
            )}
            <span aria-hidden="true" className="pointer-events-none absolute top-full left-1/2 z-50 mt-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
              {isCollapsed ? t('sidebar.expandSidebar') : t('sidebar.collapseSidebar')}
            </span>
          </button>
        )}
      </div>

      <div className="p-4 pb-2">
        {/* address rule 2.5.8 Target Size (Minimum) */}
        <button
          onClick={handleNewChat}
          className={`group relative flex items-center gap-2 rounded-lg bg-teal-600 transition-colors hover:bg-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1 ${
            isCollapsed ? 'w-full justify-center px-0 py-3' : 'w-full px-4 py-3'
          }`}
          aria-label={t('sidebar.newChat')}
        >
          <Plus className="h-4 w-4 flex-shrink-0" />
          {!isCollapsed && t('sidebar.newChat')}
          <span aria-hidden="true" className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
            {t('sidebar.newChat')}
          </span>
        </button>
      </div>

      <div className="px-4 pb-2">
        {isCollapsed ? (
          <button
            onClick={() => setCollapsed(false)}
            aria-label={t('sidebar.searchChats')}
            className="group relative flex w-full justify-center rounded-lg py-2 transition-colors hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
          >
            <Search className="h-4 w-4 text-gray-400" />
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
              {t('sidebar.searchChats')}
            </span>
          </button>
        ) : (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-500" />
            {/* address rule 3.3.2 Labels or Instructions */}
            <input
              type="text"
              value={sidebarSearch}
              onChange={(e) => setSidebarSearch(e.target.value)}
              placeholder={t('sidebar.searchChatsPlaceholder')}
              aria-label={t('sidebar.searchChats')}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 pl-8 pr-7 text-sm text-white placeholder-gray-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500"
            />
            {sidebarSearch && (
              <button
                onClick={() => setSidebarSearch('')}
                aria-label={t('sidebar.clearSearch')}
                className="group absolute right-2 top-1/2 -translate-y-1/2 rounded p-1.5 transition-colors hover:bg-gray-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
              >
                <X className="h-3 w-3 text-gray-500" />
                <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
                  {t('sidebar.clearSearch')}
                </span>
              </button>
            )}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4">
        <div className="space-y-2">
          {!isCollapsed && <div className="px-4 py-2 text-sm text-gray-400">{t('sidebar.history')}</div>}

          {isConversationsLoading && !isCollapsed && (
            <div className="px-4 py-3 text-center text-sm text-gray-500">Loading...</div>
          )}

          {filteredConversations.map((conversation) => (
            <div key={conversation.id} className="group/item relative w-full">
              <button
                onClick={() => handleSwitchConversation(conversation.id)}
                className={`w-full rounded-lg text-left transition-colors hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 ${
                  currentConversationId === conversation.id ? 'bg-gray-800' : ''
                } ${isCollapsed ? 'flex justify-center px-0 py-3' : 'px-4 py-3 pr-10'}`}
                aria-label={conversation.title}
                aria-current={currentConversationId === conversation.id ? 'true' : undefined}
              >
                <div className="flex items-center gap-2">
                  {isConversationLoading(conversation.id) ? (
                    <span className="h-4 w-4 flex-shrink-0 animate-spin rounded-full border-2 border-teal-400 border-t-transparent" aria-label="Loading" />
                  ) : (
                    <MessageSquare className="h-4 w-4 flex-shrink-0 text-teal-400" />
                  )}
                  {!isCollapsed && <span className="truncate">{conversation.title}</span>}
                </div>
                {isCollapsed && (
                  <span aria-hidden="true" className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover/item:opacity-100">
                    {conversation.title}
                  </span>
                )}
              </button>

              {!isCollapsed && (
                <button
                  onClick={(e) => handleDeleteConversation(e, conversation.id)}
                  aria-label={`Delete conversation: ${conversation.title}`}
                  className="group/del absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 opacity-0 transition-opacity group-hover/item:opacity-100 hover:bg-gray-700 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
                >
                  <Trash2 className="h-3.5 w-3.5 text-gray-500 hover:text-red-400" />
                  <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover/del:opacity-100 group-focus-visible/del:opacity-100">
                    {t('sidebar.deleteChat')}
                  </span>
                </button>
              )}
            </div>
          ))}

          {!isCollapsed && sidebarSearch && filteredConversations.length === 0 && (
            <div className="px-4 py-3 text-center text-sm text-gray-500">{t('sidebar.noResults')}</div>
          )}

          {!isCollapsed && !isConversationsLoading && conversations.length === 0 && !sidebarSearch && (
            <div className="px-4 py-3 text-center text-sm text-gray-500">No chats yet</div>
          )}
        </div>
      </div>

      <div className="border-t border-gray-800 p-4 dark:border-gray-800">
        {isCollapsed ? (
          <div className="mb-3 flex justify-center" aria-label={getStatusText()} role="status">
            <Circle aria-hidden="true" className={`h-2 w-2 fill-current ${getStatusColor()}`} />
          </div>
        ) : (
          <div className="mb-3 rounded-lg bg-gray-800 px-4 py-2 dark:bg-gray-800">
            <div className="flex items-center gap-2 text-xs">
              <Circle className={`h-2 w-2 fill-current ${getStatusColor()}`} />
              <span className="text-gray-400 dark:text-gray-400">{getStatusText()}</span>
            </div>
          </div>
        )}

        <button
          onClick={async () => {
            await clearChat();
          }}
          className={`group relative flex items-center gap-2 rounded-lg text-red-400 transition-colors hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 dark:text-red-400 dark:hover:bg-gray-800 ${
            isCollapsed ? 'w-full justify-center px-0 py-3' : 'w-full px-4 py-3'
          }`}
          aria-label={t('sidebar.clearChat')}
        >
          <Trash2 className="h-4 w-4 flex-shrink-0" />
          {!isCollapsed && t('sidebar.clearChat')}
          <span aria-hidden="true" className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
            {t('sidebar.clearChat')}
          </span>
        </button>
      </div>
    </div>
  );
}
