import { useState } from 'react';
import { ChatHeader } from './ChatHeader';
import { ChatInfoModal } from './ChatInfoModal';
import { ChatInput } from './ChatInput';
import { ChatMessageList } from './ChatMessageList';
import { ChatSettingsModal } from './ChatSettingsModal';

export function ChatPanel() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900 lg:border-r">
      <ChatHeader
        onOpenInfo={() => setIsInfoOpen(true)}
      />
      <ChatInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)} />
      <ChatSettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
      <ChatMessageList />
      <ChatInput onOpenSettings={() => setIsSettingsOpen(true)} />
    </div>
  );
}
