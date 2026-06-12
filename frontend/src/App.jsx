import { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatArea from './components/ChatArea.jsx'
import InputBar from './components/InputBar.jsx'
import Header from './components/Header.jsx'
import DataPanel from './components/DataPanel.jsx'
import TrainingPanel from './components/TrainingPanel.jsx'
import './styles/app.css'

const API = ''  // proxied via vite to localhost:8000

export default function App() {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem('theme') || 'dark'
  )
  const [mode, setMode] = useState('chat')          // 'chat' | 'training' | 'pipeline'
  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const abortRef = useRef(null)

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      const res = await fetch(`${API}/conversations`)
      const data = await res.json()
      setConversations(data.conversations || [])
    } catch {}
  }

  const startNewConversation = async () => {
    if (isStreaming && abortRef.current) abortRef.current.abort()
    try {
      const res = await fetch(`${API}/conversations/new`, { method: 'POST' })
      const data = await res.json()
      setActiveConvId(data.conversation_id)
      setMessages([])
    } catch {
      const id = crypto.randomUUID()
      setActiveConvId(id)
      setMessages([])
    }
  }

  const loadConversation = async (convId) => {
    setActiveConvId(convId)
    try {
      const res = await fetch(`${API}/conversations/${convId}`)
      const data = await res.json()
      const msgs = (data.messages || []).flatMap(m => [
        { id: m.timestamp + '-q', role: 'user', content: m.query },
        {
          id: m.timestamp + '-a',
          role: 'assistant',
          content: m.answer,
          intent: m.intent,
          sources: m.sources || [],
          timing: m.timing || {},
        }
      ])
      setMessages(msgs)
    } catch {}
  }

  const deleteConversation = async (convId) => {
    try {
      await fetch(`${API}/conversations/${convId}`, { method: 'DELETE' })
      if (convId === activeConvId) {
        setActiveConvId(null)
        setMessages([])
      }
      loadConversations()
    } catch {}
  }

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || isStreaming) return

    let convId = activeConvId
    if (!convId) {
      convId = crypto.randomUUID()
      setActiveConvId(convId)
    }

    const userMsg      = { id: Date.now() + '-u', role: 'user', content: query }
    const assistantMsg = { id: Date.now() + '-a', role: 'assistant', content: '', intent: null, sources: [], timing: {}, streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, conversation_id: convId }),
        signal: controller.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try { handleSSEEvent(JSON.parse(line.slice(6)), assistantMsg.id) } catch {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: 'Connection error. Is the API running?', streaming: false }
            : m
        ))
      }
    } finally {
      setIsStreaming(false)
      setMessages(prev => prev.map(m => m.id === assistantMsg.id ? { ...m, streaming: false } : m))
      loadConversations()
    }
  }, [activeConvId, isStreaming])

  const handleSSEEvent = (event, msgId) => {
    switch (event.type) {
      case 'intent':
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, intent: event.intent } : m))
        break
      case 'token':
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: m.content + event.content } : m))
        break
      case 'metadata':
        setMessages(prev => prev.map(m =>
          m.id === msgId ? { ...m, intent: event.intent, sources: event.sources || [], timing: event.timing || {} } : m
        ))
        break
      case 'error':
        setMessages(prev => prev.map(m =>
          m.id === msgId ? { ...m, content: event.content || 'Error', streaming: false } : m
        ))
        break
    }
  }

  const stopStreaming = () => {
    if (abortRef.current) abortRef.current.abort()
    setIsStreaming(false)
    setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m))
  }

  return (
    <div className={`app-layout ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <Sidebar
        conversations={conversations}
        activeConvId={activeConvId}
        onNew={startNewConversation}
        onSelect={(id) => { loadConversation(id); setMode('chat') }}
        onDelete={deleteConversation}
        isOpen={sidebarOpen}
      />
      <div className="main-area">
        <Header
          theme={theme}
          onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          onToggleSidebar={() => setSidebarOpen(o => !o)}
          sidebarOpen={sidebarOpen}
          mode={mode}
          onToggleMode={() => setMode(m => m === 'chat' ? 'training' : m === 'training' ? 'pipeline' : 'chat')}
          onSetMode={setMode}
        />

        {mode === 'chat' ? (
          <>
            <ChatArea messages={messages} isStreaming={isStreaming} />
            <InputBar
              onSend={sendMessage}
              onStop={stopStreaming}
              isStreaming={isStreaming}
              disabled={false}
            />
          </>
        ) : mode === 'training' ? (
          <DataPanel />
        ) : (
          <TrainingPanel />
        )}
      </div>
    </div>
  )
}
