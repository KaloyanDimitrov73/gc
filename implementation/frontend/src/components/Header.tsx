import { useEffect, useRef, useState } from 'react';
import { Bot} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../contexts/AuthContext';
import {
  Moon,
  Sun,
  Globe,
  LogOut,
  User,
  LogIn,
  Gitlab,
  FileText,
  Menu,
  MoreVertical,
  X,
  ChevronRight,
  ChevronLeft,
  Info,
} from 'lucide-react';
import logoDark from '../assets/nfdixcs_logo.svg';
import logoLight from '../assets/nfdixcs_logo_white.svg';
import { ChatInfoModal } from './chat/ChatInfoModal';

interface HeaderProps {
  onMenuClick: () => void;
  isDesktop: boolean;
}

export function Header({ onMenuClick, isDesktop }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();
  const { user, logout, isLoading } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showAboutModal, setShowAboutModal] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showQuickLinks, setShowQuickLinks] = useState(false);
  const [showQuickLanguage, setShowQuickLanguage] = useState(false);
  const actionMenusRef = useRef<HTMLDivElement>(null);

  const languages: Array<{ code: 'en' | 'de'; name: string }> = [
    { code: 'en', name: 'English' },
    { code: 'de', name: 'Deutsch' },
  ];

  useEffect(() => {
    const onMouseDown = (event: MouseEvent) => {
      if (
        actionMenusRef.current &&
        !actionMenusRef.current.contains(event.target as Node)
      ) {
        setShowUserMenu(false);
        setShowQuickLinks(false);
      }
    };

    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  return (
    <>
      <header className="border-b border-gray-200 bg-[#E5E7EB] px-3 py-3 dark:border-gray-800 dark:bg-gray-900 sm:px-4 lg:px-6 lg:py-4">
        <div className="flex items-center justify-between gap-2 sm:gap-3 lg:gap-6">
          <div className="flex min-w-0 items-center gap-2 sm:gap-3 lg:gap-4">
            {/* address rule 2.5.2 Pointer Cancellation */}
            {/* address rule 2.5.8 Target Size (Minimum) */}
            {/* address rule 1.4.11 Non-text Contrast */}
            {!isDesktop && (
              <button
                onClick={onMenuClick}
                className="group relative rounded-lg p-2 text-gray-700 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-gray-200 dark:hover:bg-gray-800"
                title="Open navigation"
                aria-label="Open navigation"
              >
                <Menu className="h-5 w-5" />
                {/* address rule 1.4.13 Content on Hover or Focus */}
                <span aria-hidden="true" className="pointer-events-none absolute top-full left-1/2 z-50 mt-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
                  Open navigation
                </span>
              </button>
            )}

            <a
                href="https://nfdixcs.org/"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="NFDIxCS – visit homepage (opens in new tab)"
            >
                <img
                    src={theme === 'dark' ? logoLight : logoDark}
                    alt="NFDIxCS Logo"
                    className="w-28 object-contain sm:w-32 md:w-36 lg:w-44 xl:w-48"
                />
            </a>

            <Bot className="mt-0.5 h-6 w-6 flex-shrink-0 text-teal-600" />
            <h1 className="font-bold text-gray-900 dark:text-white lg:text-xl">
               SciGraphChat
            </h1>
            <span className="hidden lg:inline"> {t('chat.subtitle')}</span>
          </div>

          <div
            ref={actionMenusRef}
            className="flex shrink-0 items-center gap-1 sm:gap-2 lg:gap-3"
          >
            <button
              onClick={() => setShowAboutModal(true)}
              className="rounded-lg px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-gray-300 dark:hover:bg-gray-800"
              aria-label="About this application"
            >
              About
            </button>
            <div className="relative">
              {/* address rule 4.1.2 Name, Role, Value */}
              <button
                onClick={() => {
                  setShowQuickLinks((value) => !value);
                  setShowUserMenu(false);
                }}
                className="group relative rounded-lg p-2 text-gray-700 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-gray-300 dark:hover:bg-gray-800"
                title="More actions"
                aria-label="More actions"
                aria-expanded={showQuickLinks}
                aria-haspopup="menu"
              >
                <MoreVertical className="h-5 w-5" />
                <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
                  More actions
                </span>
              </button>
              {showQuickLinks && (
                <div role="menu" className="absolute right-0 mt-2 w-48 rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-gray-700 dark:bg-gray-800 z-50">
                  {showQuickLanguage ? (
                    <>
                      <button
                        onClick={() => setShowQuickLanguage(false)}
                        role="menuitem"
                        className="flex w-full items-center gap-2 px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                      >
                        <ChevronLeft className="h-4 w-4" />
                        <span>{t('header.language')}</span>
                      </button>
                      <div className="my-1 border-t border-gray-200 dark:border-gray-700" />
                      {/* address rule 1.4.1 Use of Color */}
                      {languages.map((lang) => (
                        <button
                          key={lang.code}
                          onClick={() => { setLanguage(lang.code); setShowQuickLinks(false); setShowQuickLanguage(false); }}
                          role="menuitemradio"
                          aria-checked={language === lang.code}
                          className={`flex w-full items-center gap-2 px-3 py-2 text-sm transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 ${
                            language === lang.code
                              ? 'font-semibold text-blue-600 dark:text-blue-400'
                              : 'text-gray-700 dark:text-gray-200'
                          }`}
                        >
                          <Globe className="h-4 w-4" />
                          <span>{lang.name}</span>
                        </button>
                      ))}
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setShowQuickLanguage(true)}
                        role="menuitem"
                        aria-haspopup="menu"
                        className="flex w-full items-center justify-between px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                      >
                        <span className="flex items-center gap-2">
                          <Globe className="h-4 w-4" />
                          <span>{t('header.language')}</span>
                        </span>
                        <span className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
                          <span className="uppercase">{language}</span>
                          <ChevronRight className="h-4 w-4" />
                        </span>
                      </button>
                      <div className="my-1 border-t border-gray-200 dark:border-gray-700" />
                      <button
                        onClick={() => { toggleTheme(); setShowQuickLinks(false); }}
                        role="menuitem"
                        className="flex w-full items-center gap-2 px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                      >
                        {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                        <span>{t('header.theme')}</span>
                      </button>
                      <div className="my-1 border-t border-gray-200 dark:border-gray-700" />
                      <a
                        href="#terms-of-use"
                        role="menuitem"
                        className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => setShowQuickLinks(false)}
                      >
                        <FileText className="h-4 w-4" />
                        <span>{t('header.termsOfUse')}</span>
                      </a>
                      <a
                        href="https://gitlab.gwdg.de/nfdixcs"
                        target="_blank"
                        rel="noopener noreferrer"
                        role="menuitem"
                        className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-200 dark:hover:bg-gray-700"
                        onClick={() => setShowQuickLinks(false)}
                      >
                        <Gitlab className="h-4 w-4" />
                        <span>{t('header.openSource')}</span>
                      </a>
                    </>
                  )}
                </div>
              )}
            </div>

            {user ? (
              <div className="relative">
                <button
                  onClick={() => {
                    setShowUserMenu((value) => !value);
                    setShowQuickLinks(false);
                  }}
                  className="group relative flex items-center gap-2 rounded-lg p-2 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-800"
                  aria-label={`User menu for ${user.name}`}
                  aria-expanded={showUserMenu}
                  aria-haspopup="menu"
                >
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-sm font-bold text-white">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <span className="hidden max-w-24 truncate text-sm font-medium text-gray-700 dark:text-gray-300 lg:inline">
                    {user.name}
                  </span>
                  <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700 lg:hidden">
                    {user.name}
                  </span>
                </button>

                {showUserMenu && (
                  <div role="menu" className="absolute right-0 mt-2 w-48 rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800 z-50">
                    <div className="border-b border-gray-200 px-4 py-3 dark:border-gray-700">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{user.name}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">{user.email}</p>
                    </div>
                    <button role="menuitem" className="flex w-full items-center gap-2 px-4 py-2 text-left text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700">
                      <User className="h-4 w-4" />
                      {t('header.profile')}
                    </button>
                    <button
                      onClick={() => {
                        logout();
                        setShowUserMenu(false);
                      }}
                      role="menuitem"
                      className="flex w-full items-center gap-2 border-t border-gray-200 px-4 py-2 text-left text-red-600 transition-colors hover:bg-gray-100 dark:border-gray-700 dark:text-red-400 dark:hover:bg-gray-700"
                    >
                      <LogOut className="h-4 w-4" />
                      {t('header.logout')}
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                disabled
                className="group relative flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-white opacity-50 cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-1"
                aria-label={t('header.login')}
              >
                <LogIn className="h-4 w-4" />
                <span className="hidden sm:inline">{t('header.login')}</span>
                <span aria-hidden="true" className="pointer-events-none absolute top-full right-0 z-50 mt-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700 sm:hidden">
                  {t('header.login')}
                </span>
              </button>
            )}
          </div>
        </div>
      </header>

      {showAuthModal && <AuthModal onClose={() => setShowAuthModal(false)} />}
      {showAboutModal && <AboutModal onClose={() => setShowAboutModal(false)} />}
    </>
  );
}

function AuthModal({ onClose }: { onClose: () => void }) {
  const { login, register, isLoading, error, clearError } = useAuth();
  const { t } = useLanguage();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();

    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, password, name);
      }
      onClose();
    } catch {
      // Error is handled by context.
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4">
      <div role="dialog" aria-modal="true" aria-labelledby="auth-modal-title" className="w-full max-w-[calc(100vw-2rem)] max-h-[90dvh] overflow-y-auto rounded-lg border border-gray-200 bg-white p-5 shadow-xl dark:border-gray-800 dark:bg-gray-900 sm:max-w-md sm:p-8">
        <div className="mb-4 flex items-center justify-between sm:mb-6">
          <h2 id="auth-modal-title" className="text-xl font-bold text-gray-900 dark:text-white sm:text-2xl">
            {isLogin ? t('auth.login') : t('auth.register')}
          </h2>
          <button
            onClick={onClose}
            className="group relative rounded-lg p-1 text-gray-500 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:text-gray-300 dark:hover:bg-gray-800"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
              Close
            </span>
          </button>
        </div>

        {error && (
          <div className="mb-4 rounded-lg bg-red-100 p-3 text-sm text-red-700 dark:bg-red-900 dark:text-red-200">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <div>
              {/* address rule 3.3.2 Labels or Instructions */}
              <label htmlFor="auth-name" className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
                {t('auth.register')}
              </label>
              <input
                id="auth-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-400"
                placeholder="John Doe"
                required={!isLogin}
              />
            </div>
          )}

          <div>
            <label htmlFor="auth-email" className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
              {t('auth.email')}
            </label>
            <input
              id="auth-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-400"
              placeholder="user@example.com"
              required
            />
          </div>

          <div>
            <label htmlFor="auth-password" className="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
              {t('auth.password')}
            </label>
            <input
              id="auth-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-400"
              placeholder="********"
              required
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full rounded-lg bg-blue-600 py-2 font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? 'Loading...' : isLogin ? t('auth.login') : t('auth.register')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {isLogin ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={() => {
                setIsLogin(!isLogin);
                clearError();
              }}
              className="font-medium text-blue-600 hover:underline dark:text-blue-400"
            >
              {isLogin ? t('auth.register') : t('auth.login')}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

function AboutModal({ onClose }: { onClose: () => void }) {
  const [showInfoModal, setShowInfoModal] = useState(false);

  return (
    <>
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 px-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-modal-title"
        className="mx-4 flex max-h-[85dvh] w-full max-w-[calc(100vw-2rem)] flex-col rounded-lg bg-white shadow-xl dark:bg-gray-800 sm:max-w-2xl"
      >
        <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-700 sm:p-6">
          <h3 id="about-modal-title" className="flex items-center gap-2 text-gray-900 dark:text-white">
            <Info className="h-5 w-5 text-teal-600" />
            About
          </h3>
          <button
            onClick={onClose}
            className="group relative rounded-lg p-1 transition-colors hover:bg-gray-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 dark:hover:bg-gray-700"
            aria-label="Close"
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            <span aria-hidden="true" className="pointer-events-none absolute bottom-full right-0 z-50 mb-1.5 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100 dark:bg-gray-700">
              Close
            </span>
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4 text-sm text-gray-700 dark:text-gray-300 sm:p-6">
          <p>
            This prototype demonstrates an early version of the Chat Assistant based on Retrieval‑Augmented Generation
            (RAG) over a scientific knowledge graph and the emerging SciGraphChat framework. It is a minimal viable
            product (MVP) currently undergoing development and is intended to showcase the core interaction concepts.
          </p>
          <p>The demo provides two main user interface components:</p>
          <ul className="ml-2 list-inside list-disc space-y-1">
            <li>
              <span className="font-semibold text-gray-900 dark:text-white">Chat Panel (Text View):</span> Enables
              conversational interaction with the system. Users can enter queries, ask questions, and receive generated
              responses grounded in retrieved scientific metadata. For each chat response, users can view the sub-graph,
              find related research, follow the links to the research artifacts (i.e., papers and supplementary
              materials), share the chat link with others, and export (Markdown, HTML, PDF) the generated answers and
              retrieved information.
            </li>
            <li>
              <span className="font-semibold text-gray-900 dark:text-white">Graph Panel (Graphical View):</span> Offers
              an interactive visual environment for exploring the knowledge graph, including an integrated graph search
              via search field and click‑based navigation.
              Users can access detailed information about nodes and relationships (edges)
              within the displayed sub‑graph with a single click, while double‑clicking
              a node expands its neighboring entities or collapses them if the node was already expanded.
              A visual exploration helps users to understand complex and
              semantic relationships more easily by presenting the underlying
              knowledge graph in an intuitive and interactive way.
            </li>
          </ul>
          <p>
            This demo is connected to a curated subset of the{' '}
            <a
              href="https://orkg.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="mx-1 text-teal-600 underline dark:text-teal-400"
            >
              Open Research Knowledge Graph (ORKG)
            </a>{' '}
            focused on software architecture research based on the dataset of the paper{' '}
            <a
              href="https://doi.org/10.1109/ICSA53651.2022.00023"
              target="_blank"
              rel="noopener noreferrer"
              className="mx-1 text-teal-600 underline dark:text-teal-400"
            >
              "Evaluation methods and replicability of software architecture research objects"
            </a>{' '}
            published at the IEEE 19th International Conference on Software Architecture (ICSA).
          </p>
          <p>
            Users can inspect relationships, navigate connected research artifacts, and gain insights through
            graph‑based representations. This demonstration focuses on illustrating how text‑based retrieval and
            graph‑based exploration can work together to support research‑oriented question answering. Additional
            features and extended functionality will follow in subsequent development stages.
          </p>
          <p>
            <button
              onClick={() => setShowInfoModal(true)}
              className="text-teal-600 underline dark:text-teal-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 rounded"
            >
              View supported question types
            </button>
          </p>
        </div>
      </div>
    </div>
    <ChatInfoModal isOpen={showInfoModal} onClose={() => setShowInfoModal(false)} />
    </>
  );
}
