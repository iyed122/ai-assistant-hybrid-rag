import { useState, useRef, useEffect } from 'react'

export default function InputBar({ onSend, onStop, isStreaming }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }, [value])

  // Listen for welcome chip clicks
  useEffect(() => {
    const handler = (e) => {
      setValue(e.detail)
      textareaRef.current?.focus()
    }
    window.addEventListener('fill-input', handler)
    return () => window.removeEventListener('fill-input', handler)
  }, [])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    const q = value.trim()
    if (!q || isStreaming) return
    onSend(q)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  return (
    <div className="input-bar-wrapper">
      <div className="input-bar">
        <textarea
          ref={textareaRef}
          className="input-textarea"
          placeholder="Ask anything about your codebase, issues, or documents…"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={false}
        />
        {isStreaming ? (
          <button className="send-btn stop-btn" onClick={onStop} title="Stop">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="4" width="16" height="16" rx="2"/>
            </svg>
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!value.trim()}
            title="Send (Enter)"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        )}
      </div>
      <div className="input-hint">Enter to send · Shift+Enter for new line</div>
    </div>
  )
}
