import { useEffect, useState } from 'react';
import { ChatPanel } from './components/chat/ChatPanel';
import { NodeGraphPanel } from './components/graph/NodeGraphPanel';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { HealthProvider } from './contexts/HealthContext';
import { SettingsProvider } from './contexts/SettingsContext';
import { ChatProvider } from './contexts/ChatContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { LanguageProvider } from './contexts/LanguageContext';
import { AuthProvider } from './contexts/AuthContext';
import { MessageSquare, Network } from 'lucide-react';
import { useViewportTier } from './hooks/useViewportTier';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from './components/ui/resizable';

type PanelView = 'chat' | 'graph';

export default function App() {
  const tier = useViewportTier();
  const isDesktop = tier === 'desktop';
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileView, setMobileView] = useState<PanelView>('chat');

  useEffect(() => {
    if (isDesktop) {
      setSidebarOpen(false);
    }
  }, [isDesktop]);

  return (
    <ThemeProvider>
      <LanguageProvider>
        <AuthProvider>
          <HealthProvider>
            <SettingsProvider>
              <ChatProvider>
                <div className="flex h-[100dvh] min-h-[100dvh] min-w-0 flex-col bg-gray-50 dark:bg-gray-950">
                  {(isDesktop || mobileView === 'chat') && (
                    <Header
                      onMenuClick={() => setSidebarOpen((previous) => !previous)}
                      isDesktop={isDesktop}
                    />
                  )}

                  <div className="relative flex min-h-0 flex-1 overflow-hidden">
                    {!isDesktop && sidebarOpen && (
                      <div
                        className="fixed inset-0 z-40 bg-black/50"
                        onClick={() => setSidebarOpen(false)}
                        aria-hidden="true"
                      />
                    )}

                    {isDesktop ? (
                      <div className="z-20 min-h-0 shrink-0">
                        <Sidebar isDrawerMode={false} />
                      </div>
                    ) : (
                      <div
                        className={`fixed inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out ${
                          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
                        }`}
                      >
                        <Sidebar isDrawerMode onRequestClose={() => setSidebarOpen(false)} />
                      </div>
                    )}

                    <main className="relative flex min-h-0 flex-1 overflow-hidden">
                      {isDesktop ? (
                        <ResizablePanelGroup
                          direction="horizontal"
                          autoSaveId="desktop-chat-graph-layout"
                          className="min-h-0 min-w-0 flex-1"
                        >
                          <ResizablePanel defaultSize={50} minSize={30}>
                            <section className="flex h-full min-h-0 min-w-0">
                              <ChatPanel />
                            </section>
                          </ResizablePanel>
                          <ResizableHandle withHandle />
                          <ResizablePanel defaultSize={50} minSize={30}>
                            <section className="flex h-full min-h-0 min-w-0">
                              <NodeGraphPanel />
                            </section>
                          </ResizablePanel>
                        </ResizablePanelGroup>
                      ) : (
                        <>
                          {/* address rule 2.4.11 Focus Not Obscured (Minimum) */}
                          <section
                            className={`${
                              mobileView === 'chat' ? 'flex' : 'hidden'
                            } min-h-0 min-w-0 flex-1 pb-16`}
                          >
                            <ChatPanel />
                          </section>
                          <section
                            className={`${
                              mobileView === 'graph' ? 'flex' : 'hidden'
                            } min-h-0 min-w-0 flex-1 pb-16`}
                          >
                            <NodeGraphPanel />
                          </section>
                          <nav className="fixed inset-x-0 bottom-0 z-30 border-t border-gray-200 bg-white pb-[calc(env(safe-area-inset-bottom)+0.25rem)] dark:border-gray-800 dark:bg-gray-900">
                            {/* address rule 1.4.1 Use of Color */}
                            {/* address rule 2.5.2 Pointer Cancellation */}
                            {/* address rule 2.5.8 Target Size (Minimum) */}
                            <div className="flex">
                              <button
                                onClick={() => setMobileView('chat')}
                                className={`flex-1 px-3 py-3 text-sm transition-colors ${
                                  mobileView === 'chat'
                                    ? 'border-t-2 border-teal-600 text-teal-600 dark:text-teal-400'
                                    : 'text-gray-500 dark:text-gray-400'
                                }`}
                              >
                                <span className="flex items-center justify-center gap-2">
                                  <MessageSquare className="h-5 w-5" />
                                  <span className="font-medium">Chat</span>
                                </span>
                              </button>
                              <button
                                onClick={() => setMobileView('graph')}
                                className={`flex-1 px-3 py-3 text-sm transition-colors ${
                                  mobileView === 'graph'
                                    ? 'border-t-2 border-teal-600 text-teal-600 dark:text-teal-400'
                                    : 'text-gray-500 dark:text-gray-400'
                                }`}
                              >
                                <span className="flex items-center justify-center gap-2">
                                  <Network className="h-5 w-5" />
                                  <span className="font-medium">Graph</span>
                                </span>
                              </button>
                            </div>
                          </nav>
                        </>
                      )}
                    </main>
                  </div>
                </div>
              </ChatProvider>
            </SettingsProvider>
          </HealthProvider>
        </AuthProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}
