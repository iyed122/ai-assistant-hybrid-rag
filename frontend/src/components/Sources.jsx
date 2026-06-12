import { useState } from 'react'

function sourceBadge(source) {
  const s = (source || '').toLowerCase()
  if (s.includes('gitlab'))      return { label: 'GitLab',     cls: 'gitlab' }
  if (s.includes('jira'))        return { label: 'Jira',       cls: 'jira' }
  if (s.includes('confluence'))  return { label: 'Confluence', cls: 'confluence' }
  return { label: 'RAG', cls: 'rag' }
}

export default function Sources({ sources }) {
  const [open, setOpen] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="sources-section">
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
          <polyline points="9 18 15 12 9 6"/>
        </svg>
        {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>

      {open && (
        <div className="sources-list">
          {sources.map((s, i) => {
            const badge = sourceBadge(s.source)
            const href = s.url || null
            const Comp = href ? 'a' : 'div'
            return (
              <Comp
                key={i}
                className="source-card"
                {...(href ? { href, target: '_blank', rel: 'noopener noreferrer' } : {})}
              >
                <span className={`source-badge ${badge.cls}`}>{badge.label}</span>
                <span className="source-title">{s.title || s.project || 'Source'}</span>
                {s.score > 0 && (
                  <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                )}
              </Comp>
            )
          })}
        </div>
      )}
    </div>
  )
}
