import { useRef, useState } from 'react';
import { ChevronDown, Send, Settings } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext';
import { useLanguage } from '../../contexts/LanguageContext';

interface ChatInputProps {
  onOpenSettings: () => void;
}

export function ChatInput({ onOpenSettings }: ChatInputProps) {
  const { sendMessage } = useChat();
  const { t } = useLanguage();

  const [input, setInput] = useState('');
  const [topicEntityId, setTopicEntityId] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isEntityIdOpen, setIsEntityIdOpen] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      sendMessage(input, topicEntityId);
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) {
        sendMessage(input, topicEntityId);
        setInput('');
      }
    }
  };

  return (
    <div className="shrink-0 border-t border-gray-200 p-4 dark:border-gray-800 md:p-5 lg:p-6">
      <form onSubmit={handleSubmit} className="space-y-3">
        {/* Topic entity toggle – above the main input row */}
        <div>
          <button
            type="button"
            onClick={() => setIsEntityIdOpen((v) => !v)}
            aria-expanded={isEntityIdOpen}
            aria-controls="entity-id-section"
            className="flex items-center gap-1 rounded py-1 text-sm text-gray-500 hover:text-teal-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-gray-400 dark:hover:text-teal-400"
          >
            <ChevronDown
              className={`h-4 w-4 transition-transform ${isEntityIdOpen ? 'rotate-180' : ''}`}
            />
            {t('chat.topicEntityId')}
          </button>
          <div id="entity-id-section" className={`w-full ${isEntityIdOpen ? 'mt-2 block' : 'hidden'}`}>
            <input
              type="text"
              value={topicEntityId}
              onChange={(e) => setTopicEntityId(e.target.value)}
              placeholder={t('chat.topicEntityIdPlaceholder')}
              aria-label={t('chat.topicEntityId')}
              disabled
              className="w-full rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
        </div>
        {/* Main row: textarea left, buttons right – top/bottom edges aligned */}
        <div className="flex items-stretch gap-2">
          <textarea
            ref={textareaRef}
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t('chat.enterMessage')}
            aria-label={t('chat.enterMessage')}
            className="flex-1 min-w-0 resize-none overflow-y-auto rounded-lg border border-gray-300 bg-white py-3 pl-4 pr-4 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-500"
          />
          {/* Right: Settings (top) + Send (bottom), same width */}
          <div className="flex shrink-0 flex-col gap-2">
            {/* address rule 2.5.8 Target Size (Minimum) */}
            <button
              type="button"
              onClick={onOpenSettings}
              className="group relative flex flex-1 w-full items-center justify-between gap-2 rounded-lg bg-teal-600 px-4 py-2 text-sm text-white transition-colors hover:bg-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
              aria-label={t('chat.settings')}
            >
              <span className="hidden lg:inline">{t('chat.settings')}</span>
              <Settings className="h-5 w-5" />
              <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
                {t('chat.settings')}
              </span>
            </button>
            <button
              type="submit"
              className="group relative flex flex-1 w-full items-center justify-between gap-2 rounded-md bg-teal-600 px-4 py-2 text-sm text-white transition-colors hover:bg-teal-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
              aria-label={t('chat.sendMessage')}
            >
              <span className="hidden lg:inline">{t('chat.send')}</span>
              <Send className="h-5 w-5" />
              <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
                {t('chat.sendMessage')}
              </span>
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
