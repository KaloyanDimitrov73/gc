import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Network,
  Info,
  Search,
  ExternalLink,
  Download,
  ChevronDown,
  FileText,
  Code,
  FileDown,
  Share2,
  Link2,
  Copy,
  Check,
  Eye,
  Pencil,
} from 'lucide-react';
import type { Message } from '../types';

interface MessageActionsProps {
  message: Message;
  allMessages: Message[];
  isSelected: boolean;
  onSelectMessage: (id: string) => void;
  onSendMessage: (content: string) => Promise<void>;
}

/** Extract DOI URLs and source indices from message content.
 *  Parses the backend's "[N] Source: <name>, <doi>" footer lines to preserve
 *  the original citation index used in the answer text. */
function extractSources(content: string): { label: string; url: string }[] {
  const sources: { label: string; url: string }[] = [];
  // Match lines appended by the backend: [N] Source: <name>, <doi>
  const sourceLineRegex = /^\[(\d+)\] Source: .+?,\s*(10\.\d{4,9}\/[^\s\n]+)/gm;
  let match: RegExpExecArray | null;
  const seen = new Set<string>();
  while ((match = sourceLineRegex.exec(content)) !== null) {
    const index = match[1];
    const doi = match[2].replace(/\.+$/, ''); // strip trailing dots
    if (!seen.has(doi)) {
      seen.add(doi);
      sources.push({ label: index, url: `https://doi.org/${doi}` });
    }
  }
  return sources;
}

/** Convert message content to Markdown string. */
function toMarkdown(message: Message): string {
  let md = `# Assistant Response\n\n${message.content}\n`;
  if (message.nodes && message.nodes.length > 0) {
    md += `\n## Retrieved Nodes\n\n`;
    message.nodes.forEach((n) => {
      md += `- **${n.label}** (${n.type}) — id: \`${n.id}\`\n`;
    });
  }
  return md;
}

/** Convert message content to a self-contained HTML string. */
function toHtml(message: Message): string {
  const escaped = message.content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br/>');
  let nodesHtml = '';
  if (message.nodes && message.nodes.length > 0) {
    nodesHtml =
      '<h2>Retrieved Nodes</h2><ul>' +
      message.nodes
        .map(
          (n) =>
            `<li><strong>${n.label}</strong> (${n.type}) — id: <code>${n.id}</code></li>`
        )
        .join('') +
      '</ul>';
  }
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Response Export</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;line-height:1.6;color:#1a1a1a}
code{background:#f3f4f6;padding:2px 6px;border-radius:4px}h1,h2{color:#0a4f4a}</style>
</head><body><h1>Assistant Response</h1><p>${escaped}</p>${nodesHtml}</body></html>`;
}

/** Trigger a browser download for a blob. */
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

type SharePermission = 'view' | 'edit';

/** Encode messages into a shareable URL with permission mode in the hash. */
function buildShareUrl(
  messages: Message[],
  upToMessageId: string,
  permission: SharePermission
): string {
  // Include all messages up to and including the target message
  const idx = messages.findIndex((m) => m.id === upToMessageId);
  const slice = idx >= 0 ? messages.slice(0, idx + 1) : messages;
  const payload = {
    messages: slice,
    permission,
    sharedAt: new Date().toISOString(),
  };
  const json = JSON.stringify(payload);
  const encoded = btoa(
    encodeURIComponent(json).replace(/%([0-9A-F]{2})/g, (_, p1) =>
      String.fromCharCode(parseInt(p1, 16))
    )
  );
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}#share=${encoded}`;
}

export function MessageActions({
  message,
  allMessages,
  isSelected,
  onSelectMessage,
  onSendMessage,
}: MessageActionsProps) {
  const [showMetadata, setShowMetadata] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const [sharePermission, setSharePermission] = useState<SharePermission>('view');
  const [shareCopied, setShareCopied] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  // Close export dropdown when clicking outside
  useEffect(() => {
    if (!exportOpen) return;
    const handler = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [exportOpen]);

  const sources = extractSources(message.content);
  const hasNodes = message.nodes && message.nodes.length > 0;

  const handleFindRelated = useCallback(() => {
    // Build a follow-up query based on the first resource node, or fall back to the first few words
    const resourceNode = message.nodes?.find((n) => n.type === 'resource');
    const topic = resourceNode
      ? resourceNode.label
      : message.content.split(/\s+/).slice(0, 8).join(' ');
    onSendMessage(`Find related research on: ${topic}`);
  }, [message, onSendMessage]);

  const handleExport = useCallback(
    (format: 'markdown' | 'html' | 'pdf') => {
      setExportOpen(false);
      if (format === 'markdown') {
        const md = toMarkdown(message);
        downloadBlob(
          new Blob([md], { type: 'text/markdown;charset=utf-8' }),
          'response.md'
        );
      } else if (format === 'html') {
        const html = toHtml(message);
        downloadBlob(
          new Blob([html], { type: 'text/html;charset=utf-8' }),
          'response.html'
        );
      } else if (format === 'pdf') {
        // Open a print-friendly HTML page in a new window for the user to save as PDF
        const html = toHtml(message);
        const win = window.open('', '_blank');
        if (win) {
          win.document.write(html);
          win.document.close();
          setTimeout(() => win.print(), 400);
        }
      }
    },
    [message]
  );

  const handleShareCopy = useCallback(async () => {
    const url = buildShareUrl(allMessages, message.id, sharePermission);
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
  }, [allMessages, message.id, sharePermission]);

  // ── Shared button style ──
  // address rule 1.4.11 Non-text Contrast
  const linkClass =
    'inline-flex items-center gap-1.5 rounded py-1 text-xs text-gray-500 dark:text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 transition-colors cursor-pointer select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500';

  return (
    <div className="mt-3 pt-3 border-t border-gray-300 dark:border-gray-600 space-y-2">
      {/* Action links row */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {/* View Retrieval Graph (only when nodes exist) */}
        {/* address rule 2.5.2 Pointer Cancellation */}
        {hasNodes && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onSelectMessage(message.id);
            }}
            aria-pressed={isSelected}
            className={linkClass}
          >
            <Network className="w-3.5 h-3.5" />
            <span>View Graph</span>
          </button>
        )}

        {/* Show Metadata */}
        {hasNodes && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMetadata((v) => !v);
            }}
            aria-expanded={showMetadata}
            aria-controls={`metadata-${message.id}`}
            className={linkClass}
          >
            <Info className="w-3.5 h-3.5" />
            <span>Show Metadata</span>
          </button>
        )}

        {/* Find Related Research (hidden when no valid answer) */}
        {!message.content.includes('is not able to retrieve a valid answer') && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleFindRelated();
            }}
            className={linkClass}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Find Related Research</span>
          </button>
        )}

        {/* View Source (only when DOIs found) */}
        {sources.length > 0 && (
          <span className="inline-flex items-center gap-1.5">
            <ExternalLink className="w-3.5 h-3.5 text-gray-500 dark:text-gray-400" />
            <span className="text-xs text-gray-500 dark:text-gray-400">Sources:</span>
            {sources.map((src, i) => (
              <a
                key={i}
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="rounded text-xs text-teal-600 underline dark:text-teal-400 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
                title={src.url}
              >
                [{src.label}]
                {i < sources.length - 1 && ' '}
              </a>
            ))}
          </span>
        )}

        {/* Share Chat Link (disabled) */}
        <span
          role="button"
          aria-disabled="true"
          tabIndex={0}
          className="inline-flex items-center gap-1.5 text-xs text-gray-300 dark:text-gray-600 cursor-not-allowed select-none"
          title="Sharing is not available in the demo version"
          aria-label="Share Chat Link (not available in demo)"
        >
          <Share2 className="w-3.5 h-3.5" />
          <span>Share Chat Link</span>
        </span>

        {/* Export dropdown */}
        <div className="relative" ref={exportRef}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              setExportOpen((v) => !v);
            }}
            aria-expanded={exportOpen}
            aria-haspopup="menu"
            className={linkClass}
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export</span>
            <ChevronDown className={`w-3 h-3 transition-transform ${exportOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* address rule 4.1.2 Name, Role, Value */}
          {exportOpen && (
            <div role="menu" className="absolute left-0 bottom-full mb-1 w-40 bg-white dark:bg-gray-700 rounded-lg shadow-lg border border-gray-200 dark:border-gray-600 py-1 z-50">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleExport('markdown');
                }}
                role="menuitem"
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
              >
                <FileText className="w-3.5 h-3.5" />
                Markdown
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleExport('html');
                }}
                role="menuitem"
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
              >
                <Code className="w-3.5 h-3.5" />
                HTML
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleExport('pdf');
                }}
                role="menuitem"
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500"
              >
                <FileDown className="w-3.5 h-3.5" />
                PDF
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Metadata panel (collapsible) */}
      {showMetadata && hasNodes && (
        <div id={`metadata-${message.id}`} className="mt-2 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <div className="flex justify-between">
            <span>Total nodes</span>
            <span className="font-medium text-gray-800 dark:text-gray-200">{message.nodes!.length}</span>
          </div>
          {Array.from(new Set(message.nodes!.map((n) => n.type))).map((type) => (
            <div key={type} className="flex justify-between">
              <span className="capitalize">{type}s</span>
              <span className="font-medium text-gray-800 dark:text-gray-200">
                {message.nodes!.filter((n) => n.type === type).length}
              </span>
            </div>
          ))}
          <div className="flex justify-between">
            <span>Total connections</span>
            <span className="font-medium text-gray-800 dark:text-gray-200">
              {message.nodes!.reduce((sum, n) => sum + n.connections.length, 0)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
