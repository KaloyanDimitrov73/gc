import { useEffect, useRef } from 'react';
import { Bot, User } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext';
import { useLanguage } from '../../contexts/LanguageContext';
import { MessageActions } from '../MessageActions';

export function ChatMessageList() {
  const { messages, isLoading, progress, currentStep, hubCompleted, hubTotal, selectedMessageId, selectMessage, sendMessage } = useChat();
  const { t } = useLanguage();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4 md:p-5 lg:p-6">
      {messages.length === 0 ? (
        <div className="flex h-full items-center justify-center text-gray-500 dark:text-gray-500">
          <div className="text-center">
            <Bot className="mx-auto mb-4 h-16 w-16 opacity-20" />
            <p>{t('chat.startConversation')}</p>
          </div>
        </div>
      ) : (
        <div className="space-y-5 lg:space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-2 sm:gap-3 lg:gap-4 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-teal-600">
                  <Bot className="h-5 w-5 text-white" />
                </div>
              )}
              <div
                className={`max-w-[85%] rounded-lg p-4 sm:max-w-[78%] lg:max-w-[70%] ${
                  message.role === 'user'
                    ? 'bg-teal-600 text-white'
                    : `bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-white ${
                        selectedMessageId === message.id ? 'ring-2 ring-teal-500' : ''
                      }`
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.role === 'assistant' && !message.isError && message.id !== 'msg-welcome' && (
                  <MessageActions
                    message={message}
                    allMessages={messages}
                    isSelected={selectedMessageId === message.id}
                    onSelectMessage={selectMessage}
                    onSendMessage={sendMessage}
                  />
                )}
              </div>
              {message.role === 'user' && (
                <div className="user-avatar flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gray-600 dark:bg-gray-500">
                  <User className="h-5 w-5 text-white" />
                </div>
              )}
            </div>
          ))}
          {/* address rule 2.2.2 Pause, Stop, Hide */}
          {isLoading && (
            <div className="flex justify-start gap-2 sm:gap-3 lg:gap-4" aria-live="polite" aria-label={t('chat.retrievingAnswer')}>
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-teal-600">
                <Bot className="h-5 w-5 animate-pulse text-white" aria-hidden="true" />
              </div>
              <div className="max-w-[85%] rounded-lg bg-gray-100 p-4 dark:bg-gray-800 sm:max-w-[78%] lg:max-w-[70%]">
                <div className="mb-2 flex items-center gap-1.5">
                  <span aria-hidden="true" className="h-2 w-2 animate-bounce rounded-full bg-teal-600 [animation-delay:-0.3s]"></span>
                  <span aria-hidden="true" className="h-2 w-2 animate-bounce rounded-full bg-teal-600 [animation-delay:-0.15s]"></span>
                  <span aria-hidden="true" className="h-2 w-2 animate-bounce rounded-full bg-teal-600"></span>
                  {/* inline-grid stacks all labels in the same cell so the container is always
                      sized to the widest possible label, preventing layout shifts */}
                  <span className="ml-1 inline-grid text-sm text-gray-700 dark:text-gray-300">
                    {(['input_validation', 'retrieving', 'preparing', 'analyzing', 'generating', 'processing', 'saving'] as const).map(step => (
                      <span key={step} aria-hidden="true" className="invisible col-start-1 row-start-1 whitespace-nowrap">
                        {t(`chat.step.${step}` as Parameters<typeof t>[0])}
                      </span>
                    ))}
                    <span aria-hidden="true" className="invisible col-start-1 row-start-1 whitespace-nowrap">
                      {t('chat.retrievingAnswer')}
                    </span>
                    <span className="col-start-1 row-start-1">
                      {currentStep
                        ? t(`chat.step.${currentStep}` as Parameters<typeof t>[0])
                        : t('chat.retrievingAnswer')}
                    </span>
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                  <div
                    className="progress-shimmer h-1.5 rounded-full transition-[width] duration-500 ease-out"
                    style={{ width: `${progress ?? 0}%` }}
                    role="progressbar"
                    aria-valuenow={progress ?? 0}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
                {progress !== null && (
                  <p className="mt-1 text-right text-xs text-gray-400 dark:text-gray-500">
                    {progress}%
                  </p>
                )}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
}
