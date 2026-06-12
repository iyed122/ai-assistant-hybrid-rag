import { useState } from 'react'

/**
 * timeAgo — converts an ISO timestamp string to a human-readable relative time.
 *
 * THE BUG this fixes:
 *   Python's datetime.isoformat() and datetime.utcnow().isoformat() both emit
 *   strings like "2026-06-02T10:00:00.123456" with NO timezone suffix.
 *   Browsers interpret a bare ISO string without a timezone as LOCAL time, not UTC.
 *   So in a UTC+2 environment, a message sent at 10:00 UTC is parsed as 10:00 local
 *   (= 08:00 UTC), making the diff 2 hours instead of 0 seconds.
 *   Fix: if the string has no timezone marker, append 'Z' to force UTC parsing.
 */
function timeAgo(isoString) {
  if (!isoString) return ''

  // Handle MongoDB extended JSON objects: { "$date": "..." }
  const raw = typeof isoString === 'object'
    ? (isoString.$date || String(isoString))
    : String(isoString)

  // Append 'Z' only when no timezone marker is present (no Z, no +HH:MM, no -HH:MM)
  const normalized = /[Zz]$|[+-]\d{2}:?\d{2}$/.test(raw) ? raw : raw + 'Z'

  const date = new Date(normalized)
  if (isNaN(date.getTime())) return ''

  const diff = Date.now() - date.getTime()
  if (diff < 5000) return 'just now'           // within 5 s — catches clock skew too
  const mins = Math.floor(diff / 60000)
  if (mins < 1)  return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24)  return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export default function Sidebar({ conversations, activeConvId, onNew, onSelect, onDelete, isOpen }) {
  const [hoveredId, setHoveredId] = useState(null)

  return (
    <div className={`sidebar ${isOpen ? '' : 'closed'}`}>
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNew}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          New conversation
        </button>
      </div>

      {conversations.length > 0 && (
        <div className="sidebar-section-label">Recent</div>
      )}

      <div className="conv-list">
        {conversations.length === 0 && (
          <div style={{ padding: '1.5rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            No conversations yet
          </div>
        )}
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`conv-item ${activeConvId === conv.id ? 'active' : ''}`}
            onClick={() => onSelect(conv.id)}
            onMouseEnter={() => setHoveredId(conv.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            <span className="conv-icon">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </span>
            <div className="conv-text">
              <div className="conv-title">{conv.title || 'Untitled'}</div>
              <div className="conv-meta">{timeAgo(conv.last_time)} · {conv.count} msg{conv.count !== 1 ? 's' : ''}</div>
            </div>
            <button
              className="conv-delete"
              onClick={e => { e.stopPropagation(); onDelete(conv.id) }}
              title="Delete"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/>
              </svg>
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
