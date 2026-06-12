import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Sources from './Sources.jsx'

const INTENT_LABELS = {
  rag:      { label: '📚 RAG',      cls: 'rag' },
  sentries: { label: '📡 Sentries', cls: 'sentries' },
  both:     { label: '🔀 Both',     cls: 'both' },
}

const EXAMPLE_QUERIES = [
  'What are the open issues in auth-service?',
  'Explain the token introspection feature',
  'Show me recent merge requests',
  'What does the RAG pipeline do?',
  'List all Jira tickets in progress',
  'How does the weaver node work?',
]

function IntentBadge({ intent }) {
  if (!intent || !INTENT_LABELS[intent]) return null
  const { label, cls } = INTENT_LABELS[intent]
  return <span className={`intent-badge ${cls}`}>{label}</span>
}

function TimingBar({ timing }) {
  if (!timing || Object.keys(timing).length === 0) return null
  const total = Object.values(timing).reduce((a, b) => a + b, 0)
  return (
    <div className="timing-bar">
      {Object.entries(timing).map(([k, v]) => (
        <span key={k} className="timing-item">
          {k}: <span>{v.toFixed(2)}s</span>
        </span>
      ))}
      <span className="timing-item">total: <span>{total.toFixed(2)}s</span></span>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div className={`message ${msg.role}`}>
      <div className="message-header">
        <div className="message-avatar">
          {isUser ? 'Y' : '🤖'}
        </div>
        <span className="message-role">{isUser ? 'You' : 'Assistant'}</span>
        {!isUser && msg.intent && <IntentBadge intent={msg.intent} />}
      </div>

      <div className="message-body">
        {isUser ? (
          <div className="message-content">{msg.content}</div>
        ) : (
          <div className={`message-content prose ${msg.streaming && !msg.content ? 'cursor-blink' : ''}`}>
            {msg.content ? (
              <span className={msg.streaming ? 'cursor-blink' : ''}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </span>
            ) : (
              <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {msg.intent ? 'Generating response…' : 'Thinking…'}
              </span>
            )}
          </div>
        )}

        {!isUser && !msg.streaming && msg.sources?.length > 0 && (
          <Sources sources={msg.sources} />
        )}

        {!isUser && !msg.streaming && (
          <TimingBar timing={msg.timing} />
        )}
      </div>
    </div>
  )
}

export default function ChatArea({ messages, isStreaming }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="chat-area">
        <div className="messages-container">
          <div className="welcome">
            <div className="welcome-logo">🤖</div>
            <h1>AI <span>Assistant</span></h1>
            <p>Unified intelligence powered by RAG, live API sentries, and LLM synthesis.</p>
            <div className="welcome-chips">
              {EXAMPLE_QUERIES.map(q => (
                <button
                  key={q}
                  className="welcome-chip"
                  onClick={() => {
                    // Dispatch a custom event that InputBar listens to
                    window.dispatchEvent(new CustomEvent('fill-input', { detail: q }))
                  }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-area">
      <div className="messages-container">
        {messages.map(msg => (
          <Message key={msg.id} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
