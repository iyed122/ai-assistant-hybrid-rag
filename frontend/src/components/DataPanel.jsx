import { useState, useEffect } from 'react'

/**
 * DataPanel — Training Data Management
 *
 * Two tabs:
 *   DPO  — shows dpo_rejected records, split-pane with chosen-half editor
 *   QLoRA — shows filtered GOLD candidates, excluded records, export
 *
 * MongoDB recommendation:
 *   knowledge_base.chat_history    — source of truth (unchanged)
 *   knowledge_base.dpo_candidates  — chosen halves, curated status
 *   QLoRA candidates are derived from chat_history on the fly (no new collection)
 *
 * Backend endpoints needed:
 *   GET  /training/dpo/candidates                   → { candidates: [...] }
 *   PUT  /training/dpo/candidates/:id/chosen        → { chosen: "..." }
 *   GET  /training/qlora/candidates                 → { candidates: [...], excluded: [...] }
 */

const API = ''

// ── Shared atoms ──────────────────────────────────────────────────────────────

const TAG_STYLE = {
  hallucination:    { bg: 'rgba(239,68,68,0.12)',    color: '#ef4444',  border: 'rgba(239,68,68,0.25)' },
  tool_misuse:      { bg: 'rgba(245,158,11,0.12)',   color: '#f59e0b',  border: 'rgba(245,158,11,0.25)' },
  format_violation: { bg: 'rgba(139,92,246,0.12)',   color: '#8b5cf6',  border: 'rgba(139,92,246,0.25)' },
  retrieval_miss:   { bg: 'rgba(59,130,246,0.12)',   color: '#3b82f6',  border: 'rgba(59,130,246,0.25)' },
}

const INTENT_STYLE = {
  rag:      { bg: 'rgba(16,185,129,0.12)',  color: '#10b981' },
  sentries: { bg: 'rgba(139,92,246,0.12)', color: '#8b5cf6' },
  both:     { bg: 'rgba(245,158,11,0.12)', color: '#f59e0b' },
}

function Tag({ label }) {
  const s = TAG_STYLE[label] || { bg: 'rgba(100,100,100,0.1)', color: '#94a3b8', border: 'rgba(100,100,100,0.2)' }
  return (
    <span style={{
      padding: '0.12rem 0.5rem', borderRadius: 999,
      fontFamily: 'var(--font-mono)', fontSize: '0.62rem', fontWeight: 600,
      textTransform: 'uppercase', letterSpacing: '0.05em',
      background: s.bg, color: s.color, border: `1px solid ${s.border || s.bg}`,
      whiteSpace: 'nowrap',
    }}>
      {label.replace('_', ' ')}
    </span>
  )
}

function IntentPill({ intent }) {
  if (!intent) return null
  const s = INTENT_STYLE[intent] || { bg: 'rgba(100,100,100,0.1)', color: '#94a3b8' }
  return (
    <span style={{
      padding: '0.1rem 0.45rem', borderRadius: 999,
      fontFamily: 'var(--font-mono)', fontSize: '0.6rem', fontWeight: 500,
      background: s.bg, color: s.color, whiteSpace: 'nowrap',
    }}>
      {intent}
    </span>
  )
}

function StatChip({ label, value, color }) {
  return (
    <span className="dp-stat">
      <strong style={color ? { color } : {}}>{value}</strong>
      <span>{label}</span>
    </span>
  )
}

function FilterPill({ label, active, onClick }) {
  return (
    <button className={`dp-filter-pill ${active ? 'active' : ''}`} onClick={onClick}>
      {label}
    </button>
  )
}

function exportJSONL(records, filename) {
  if (!records.length) { alert('Nothing to export with this filter.'); return }
  const lines = records.map(r => JSON.stringify(r)).join('\n')
  const blob = new Blob([lines], { type: 'application/jsonlines' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── DPO Panel ─────────────────────────────────────────────────────────────────

function DPOPanel() {
  const [records,    setRecords]    = useState([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [chosen,     setChosen]     = useState('')
  const [saving,     setSaving]     = useState(false)
  const [saved,      setSaved]      = useState(false)
  const [filter,     setFilter]     = useState('all')  // all | pending | curated | hallucination | tool_misuse

  useEffect(() => {
    fetch(`${API}/training/dpo/candidates`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => { setRecords(d.candidates || []); setLoading(false) })
      .catch(() => {
        setError('Could not reach GET /training/dpo/candidates — add this endpoint to your backend.')
        setLoading(false)
      })
  }, [])

  const selected = records.find(r => r.id === selectedId)

  // Sync chosen text when selection changes
  useEffect(() => {
    if (selected) { setChosen(selected.chosen || ''); setSaved(false) }
  }, [selectedId])

  const saveChosen = async () => {
    if (!selectedId || chosen.trim().length < 50) return
    setSaving(true)
    try {
      const res = await fetch(`${API}/training/dpo/candidates/${selectedId}/chosen`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chosen: chosen.trim() }),
      })
      if (!res.ok) throw new Error(res.status)
      setRecords(prev => prev.map(r =>
        r.id === selectedId ? { ...r, chosen: chosen.trim(), curated: true } : r
      ))
      setSaved(true)
    } catch { alert('Failed to save — check the backend endpoint.') }
    setSaving(false)
  }

  const filtered = records.filter(r => {
    if (filter === 'pending')       return !r.curated
    if (filter === 'curated')       return r.curated
    if (filter === 'hallucination') return (r.failure_tags || []).includes('hallucination')
    if (filter === 'tool_misuse')   return (r.failure_tags || []).includes('tool_misuse')
    return true
  })

  const curated = records.filter(r => r.curated).length
  const pending  = records.length - curated

  const handleExport = () => {
    const pairs = records
      .filter(r => r.chosen && r.chosen.trim().length >= 50)
      .map(r => ({ prompt: r.query, rejected: r.rejected, chosen: r.chosen }))
    exportJSONL(pairs, 'dpo_train.jsonl')
  }

  return (
    <div className="dp-split">
      {/* ── Left: list ── */}
      <div className="dp-list-pane">
        <div className="dp-stats-row">
          <StatChip value={records.length} label="total" />
          <StatChip value={curated}  label="curated"  color="var(--intent-rag)" />
          <StatChip value={pending}  label="pending"  color="var(--intent-both)" />
          <button className="dp-export-btn" onClick={handleExport} disabled={curated === 0}>
            Export JSONL
          </button>
        </div>

        <div className="dp-filter-row">
          {[
            { key: 'all',           label: 'All' },
            { key: 'pending',       label: 'Pending' },
            { key: 'curated',       label: 'Curated' },
            { key: 'hallucination', label: 'Hallucination' },
            { key: 'tool_misuse',   label: 'Tool Misuse' },
          ].map(f => (
            <FilterPill key={f.key} label={f.label} active={filter === f.key} onClick={() => setFilter(f.key)} />
          ))}
        </div>

        {loading && <div className="dp-state-msg">Loading candidates…</div>}
        {error   && <div className="dp-error">{error}</div>}

        <div className="dp-candidate-list">
          {filtered.map(r => (
            <button
              key={r.id}
              className={`dp-candidate-item ${selectedId === r.id ? 'active' : ''} ${r.curated ? 'curated' : ''}`}
              onClick={() => setSelectedId(r.id)}
            >
              <div className="dp-candidate-row">
                <span className={r.curated ? 'dp-dot-curated' : 'dp-dot-pending'}>
                  {r.curated ? '✓' : '○'}
                </span>
                <div className="dp-candidate-tags">
                  {(r.failure_tags || []).map(t => <Tag key={t} label={t} />)}
                </div>
                <IntentPill intent={r.intent} />
              </div>
              <div className="dp-candidate-query">{r.query}</div>
              <div className="dp-candidate-meta">
                score {r.weighted_score?.toFixed(3)} · faith {r.faithfulness?.toFixed(2)}
              </div>
            </button>
          ))}
          {!loading && !error && filtered.length === 0 && (
            <div className="dp-state-msg">No records match this filter.</div>
          )}
        </div>
      </div>

      {/* ── Right: editor ── */}
      <div className="dp-editor-pane">
        {!selected ? (
          <div className="dp-editor-empty">
            <div className="dp-editor-empty-icon">📋</div>
            <p>Select a candidate to review its rejected answer and write the correct chosen half.</p>
            <p style={{ fontSize: '0.78rem', marginTop: '0.5rem' }}>
              Priority: <strong>hallucination</strong> cases first — the correct answer is structurally clear:<br />
              acknowledge the ticket was not found, state what IS available, suggest where to look.
            </p>
          </div>
        ) : (
          <>
            <div className="dp-section">
              <div className="dp-section-label">Query</div>
              <div className="dp-query-box">{selected.query}</div>
            </div>

            <div className="dp-section">
              <div className="dp-section-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Rejected Answer</span>
                <div style={{ display: 'flex', gap: '0.35rem' }}>
                  {(selected.failure_tags || []).map(t => <Tag key={t} label={t} />)}
                </div>
              </div>
              {selected.failure_reason && (
                <div className="dp-failure-reason">{selected.failure_reason}</div>
              )}
              <div className="dp-rejected-box">{selected.rejected}</div>
            </div>

            <div className="dp-section dp-section-grow">
              <div className="dp-section-label">
                Chosen Answer
                <span className="dp-section-note">— never copy from the base model</span>
              </div>
              <textarea
                className="dp-chosen-textarea"
                placeholder={`Write the correct answer here.\n\nFor hallucination: "Ticket ${selected.query?.match(/[A-Z]{2,5}-\d+/)?.[0] || 'X'} was not found in the retrieved data. The closest available information is [...]"\n\nFor tool_misuse: cite the retrieved key explicitly and use its content.`}
                value={chosen}
                onChange={e => { setChosen(e.target.value); setSaved(false) }}
              />
              <div className="dp-chosen-footer">
                <span className={`dp-char-count ${chosen.length > 0 && chosen.length < 50 ? 'warn' : ''}`}>
                  {chosen.length} chars{chosen.length > 0 && chosen.length < 50 ? ' · min 50' : ''}
                </span>
                <button
                  className={`dp-save-btn ${saved ? 'saved' : ''}`}
                  onClick={saveChosen}
                  disabled={saving || chosen.trim().length < 50}
                >
                  {saving ? 'Saving…' : saved ? '✓ Saved' : 'Save chosen half'}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── QLoRA Panel ───────────────────────────────────────────────────────────────

const EXCLUSION_LABELS = {
  no_live_sentries:                   'no live sentries data',
  no_real_context:                    'scored against empty context',
  no_keys_cited_despite_jira_sources: 'no keys cited (generic prose)',
  manually_excluded:                  'manually excluded',
}

function QLoRAPanel() {
  const [candidates, setCandidates] = useState([])
  const [excluded,   setExcluded]   = useState([])
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [filter,     setFilter]     = useState('all')   // all | rag | both | sentries
  const [showExcl,   setShowExcl]   = useState(false)

  useEffect(() => {
    fetch(`${API}/training/qlora/candidates`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => { setCandidates(d.candidates || []); setExcluded(d.excluded || []); setLoading(false) })
      .catch(() => {
        setError('Could not reach GET /training/qlora/candidates — add this endpoint to your backend.')
        setLoading(false)
      })
  }, [])

  const intentCounts = candidates.reduce((acc, c) => {
    acc[c.intent] = (acc[c.intent] || 0) + 1; return acc
  }, {})

  const balanceOk = (intentCounts.sentries || 0) >= 35
                 && (intentCounts.rag      || 0) >= 25
                 && (intentCounts.both     || 0) >= 25

  const shown = filter === 'all'
    ? candidates
    : candidates.filter(c => c.intent === filter)

  const handleExport = () => {
    const rows = shown.map(c => ({ instruction: c.query, input: '', output: c.answer }))
    exportJSONL(rows, `qlora_${filter === 'all' ? 'train' : filter}.jsonl`)
  }

  const handleDelete = async (id) => {
    try {
      const res = await fetch(`${API}/training/qlora/candidates/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error(res.status)
      // Move from candidates → excluded locally so no refetch needed
      const rec = candidates.find(c => c.id === id)
      if (rec) {
        setCandidates(prev => prev.filter(c => c.id !== id))
        setExcluded(prev => [{
          ...rec,
          exclusion_reasons: ['manually_excluded'],
        }, ...prev])
      }
    } catch { alert('Delete failed — check the backend.') }
  }

  const handleRestore = async (id) => {
    try {
      const res = await fetch(`${API}/training/qlora/candidates/${id}/restore`, { method: 'POST' })
      if (!res.ok) throw new Error(res.status)
      const rec = excluded.find(c => c.id === id)
      if (rec) {
        setExcluded(prev => prev.filter(c => c.id !== id))
        setCandidates(prev => [{ ...rec, exclusion_reasons: undefined }, ...prev])
      }
    } catch { alert('Restore failed — check the backend.') }
  }

  return (
    <div className="dp-full-pane">
      <div className="dp-stats-row">
        <StatChip value={candidates.length} label="candidates" />
        <StatChip value={excluded.length}   label="excluded" color="var(--text-muted)" />
        {Object.entries(intentCounts).map(([k, v]) => (
          <span key={k} className="dp-stat" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <IntentPill intent={k} /> <strong>{v}</strong>
          </span>
        ))}
        {!balanceOk && candidates.length > 0 && (
          <span className="dp-balance-warn">⚠ not balanced — don't train yet</span>
        )}
        <button className="dp-export-btn" onClick={handleExport} disabled={shown.length === 0}>
          Export JSONL
        </button>
      </div>

      <div className="dp-filter-row">
        {[
          { key: 'all',      label: 'All' },
          { key: 'rag',      label: 'RAG' },
          { key: 'both',     label: 'Synthesis (both)' },
          { key: 'sentries', label: 'Sentries' },
        ].map(f => (
          <FilterPill key={f.key} label={f.label} active={filter === f.key} onClick={() => setFilter(f.key)} />
        ))}
      </div>

      {loading && <div className="dp-state-msg">Loading candidates…</div>}
      {error   && <div className="dp-error">{error}</div>}

      {!loading && !error && (
        <>
          <div className="qlora-table-wrapper">
            <table className="qlora-table">
              <thead>
                <tr>
                  <th>Score</th><th>Faith</th><th>Intent</th><th>Method</th><th>Query</th><th></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((c, i) => (
                  <tr key={c.id || i}>
                    <td className="qlora-mono">{c.weighted_score?.toFixed(3)}</td>
                    <td className="qlora-mono">{c.faithfulness?.toFixed(2)}</td>
                    <td><IntentPill intent={c.intent} /></td>
                    <td className="qlora-method">{c.scoring_method}</td>
                    <td className="qlora-query">{c.query}</td>
                    <td>
                      <button
                        onClick={() => handleDelete(c.id)}
                        title="Exclude from training"
                        style={{
                          background: 'none', border: '1px solid rgba(239,68,68,0.25)',
                          borderRadius: 4, padding: '0.2rem 0.35rem', cursor: 'pointer',
                          color: '#ef4444', opacity: 0.6, lineHeight: 1,
                          transition: 'opacity 0.15s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.opacity = 1}
                        onMouseLeave={e => e.currentTarget.style.opacity = 0.6}
                      >
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6"/>
                          <path d="M19 6l-1 14H6L5 6"/>
                          <path d="M10 11v6M14 11v6"/>
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
                {shown.length === 0 && (
                  <tr>
                    <td colSpan={5} className="dp-state-msg" style={{ padding: '2rem' }}>
                      No candidates in this filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Excluded accordion */}
          {excluded.length > 0 && (
            <div className="qlora-excluded">
              <button className="sources-toggle" onClick={() => setShowExcl(o => !o)}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                  style={{ transform: showExcl ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                  <polyline points="9 18 15 12 9 6"/>
                </svg>
                {excluded.length} excluded (not safe for training)
              </button>

              {showExcl && (
                <table className="qlora-table" style={{ opacity: 0.55, marginTop: '0.5rem' }}>
                  <thead>
                    <tr><th>Score</th><th>Intent</th><th>Excluded because</th><th>Query</th><th></th></tr>
                  </thead>
                  <tbody>
                    {excluded.map((r, i) => (
                      <tr key={r.id || i}>
                        <td className="qlora-mono">{r.weighted_score?.toFixed(3)}</td>
                        <td><IntentPill intent={r.intent} /></td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: '#ef4444' }}>
                          {(r.exclusion_reasons || [])
                            .map(k => EXCLUSION_LABELS[k.replace(/\(.*\)/, '').trim()] || k)
                            .join(', ')}
                        </td>
                        <td className="qlora-query" style={{ color: 'var(--text-muted)' }}>{r.query}</td>
                        <td>
                          {(r.exclusion_reasons || []).includes('manually_excluded') && (
                            <button
                              onClick={() => handleRestore(r.id)}
                              title="Restore to candidates"
                              style={{
                                background: 'none', border: '1px solid rgba(16,185,129,0.3)',
                                borderRadius: 4, padding: '0.2rem 0.4rem', cursor: 'pointer',
                                color: '#10b981', opacity: 0.7, fontSize: '0.75rem',
                                lineHeight: 1, transition: 'opacity 0.15s',
                              }}
                              onMouseEnter={e => e.currentTarget.style.opacity = 1}
                              onMouseLeave={e => e.currentTarget.style.opacity = 0.7}
                            >
                              ↩
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Main DataPanel ─────────────────────────────────────────────────────────────

export default function DataPanel() {
  const [tab, setTab] = useState('dpo')

  return (
    <div className="data-panel">
      <div className="dp-header">
        <div className="dp-header-left">
          <span className="dp-title">Training Data</span>
          <div className="dp-tabs">
            <button className={`dp-tab ${tab === 'dpo' ? 'active' : ''}`} onClick={() => setTab('dpo')}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
              DPO Candidates
            </button>
            <button className={`dp-tab ${tab === 'qlora' ? 'active' : ''}`} onClick={() => setTab('qlora')}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              QLoRA Candidates
            </button>
          </div>
        </div>
        <span className="dp-db-label">
          knowledge_base.dpo_candidates · knowledge_base.chat_history
        </span>
      </div>

      <div className="dp-content">
        {tab === 'dpo' ? <DPOPanel /> : <QLoRAPanel />}
      </div>
    </div>
  )
}
