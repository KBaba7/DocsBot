import { useState, useEffect, useRef, useCallback } from 'react'
import { marked } from 'marked'

marked.setOptions({ gfm: true, breaks: true })

// ── helpers ──────────────────────────────────────────────────────────────────

const safeJson = async (res) => {
  try { return await res.json() } catch { return { detail: 'Unexpected non-JSON response' } }
}
const prettyError = (body) => {
  if (!body) return 'Request failed.'
  if (typeof body.detail === 'string') return body.detail
  if (typeof body.message === 'string') return body.message
  return 'Request failed.'
}

const escapeHtml = (v) =>
  v.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')

const sanitizeHtml = (html) => {
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, 'text/html')
  doc.querySelectorAll('script,style,iframe,object,embed').forEach(n => n.remove())
  for (const el of doc.querySelectorAll('*')) {
    for (const attr of [...el.attributes]) {
      const name = attr.name.toLowerCase()
      const value = attr.value.toLowerCase()
      if (name.startsWith('on')) el.removeAttribute(attr.name)
      if ((name === 'href' || name === 'src') && value.startsWith('javascript:'))
        el.removeAttribute(attr.name)
    }
  }
  return doc.body.innerHTML
}

const normalizeTabular = (text) => {
  const lines = text.split('\n')
  const out = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.includes('\t')) { out.push(line); i++; continue }
    const rows = []
    while (i < lines.length && lines[i].includes('\t')) {
      rows.push(lines[i].split('\t').map(c => c.trim()))
      i++
    }
    if (rows.length < 2) { out.push(...rows.map(r => r.join(' '))); continue }
    const cols = Math.max(...rows.map(r => r.length))
    const padded = rows.map(r => {
      const c = [...r]
      while (c.length < cols) c.push('')
      return c.map(x => x.replaceAll('|','\\|').replaceAll('\n','<br>'))
    })
    out.push(`| ${padded[0].join(' | ')} |`)
    out.push(`| ${Array(cols).fill('---').join(' | ')} |`)
    for (const row of padded.slice(1)) out.push(`| ${row.join(' | ')} |`)
  }
  return out.join('\n')
}

const renderMarkdown = (text) => {
  const normalized = normalizeTabular(text || '')
  const html = marked.parse(normalized)
  return sanitizeHtml(html)
}

const stripSourceLines = (text) => {
  if (!text) return ''
  const cleaned = text.replace(/\n+\s*(?:#+\s*)?sources\s*:?\s*\n[\s\S]*$/i, '').trim()
  return cleaned.split('\n')
    .filter(l => !/^\s*no sources were used for this response\.?\s*$/i.test(l.trim()))
    .filter(l => !/^\s*no citations available for this turn\.?\s*$/i.test(l.trim()))
    .join('\n').trim()
}

// ── sub-components ────────────────────────────────────────────────────────────

function SourcesPanel({ sources }) {
  const vector = sources?.vector || []
  const web = sources?.web || []
  if (!vector.length && !web.length)
    return <p className="muted">No citations available for this turn.</p>
  return (
    <div className="sources-panel">
      {vector.length > 0 && (
        <section>
          <h4>Document Sources</h4>
          <div className="source-list">
            {vector.map((src, i) => {
              const pageNum = parseInt(src.page || '', 10)
              const anchor = isFinite(pageNum) && pageNum > 0 ? `#page=${pageNum}` : ''
              const pdfUrl = src.document_id
                ? `/documents/${encodeURIComponent(src.document_id)}/pdf${anchor}`
                : ''
              return (
                <article className="source-card" key={i}>
                  <div className="source-meta">
                    <span className="source-doc">{src.document || 'Unknown document'}</span>
                    <div>
                      <span className="source-page">Page {src.page || 'unknown'}</span>
                    </div>
                  </div>
                  <p className="source-excerpt">{src.excerpt || 'No excerpt available.'}</p>
                  {pdfUrl && (
                    <div className="source-actions">
                      <a className="source-link" href={pdfUrl} target="_blank" rel="noopener noreferrer">
                        Open PDF
                      </a>
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        </section>
      )}
      {web.length > 0 && (
        <section>
          <h4>Web Sources</h4>
          <ul className="source-web-list">
            {web.map((src, i) =>
              src.url && src.url !== 'N/A'
                ? <li key={i}><a href={src.url} target="_blank" rel="noopener noreferrer">{src.title || 'Untitled source'}</a></li>
                : <li key={i}>{src.title || 'Untitled source'}</li>
            )}
          </ul>
        </section>
      )}
    </div>
  )
}

function AssistantBubble({ text, sources, pending, isError }) {
  const answerText = stripSourceLines(text) || 'Response received.'
  if (pending) return (
    <div className="chat-bubble chat-bubble-assistant chat-pending">{text}</div>
  )
  if (isError) return (
    <div className="chat-bubble chat-bubble-assistant chat-error">{text}</div>
  )
  return (
    <div className="chat-bubble chat-bubble-assistant chat-markdown">
      <div className="message-panel" dangerouslySetInnerHTML={{ __html: renderMarkdown(answerText) }} />
      {sources !== undefined && (
        <details className="source-dropdown">
          <summary>Sources</summary>
          <SourcesPanel sources={sources} />
        </details>
      )}
    </div>
  )
}

function DocCard({ doc, onDelete }) {
  return (
    <article className="doc">
      <header className="doc-head">
        <h3>{doc.filename}</h3>
        <span className="doc-pages">{doc.page_count} pages</span>
      </header>
      <div className="doc-actions">
        <button className="danger" onClick={() => onDelete(doc)}>Delete</button>
      </div>
    </article>
  )
}

// ── auth page ─────────────────────────────────────────────────────────────────

function AuthPage({ onAuth, dbUnavailable }) {
  const [regEmail, setRegEmail] = useState('')
  const [regPass, setRegPass] = useState('')
  const [regMsg, setRegMsg] = useState('')
  const [regErr, setRegErr] = useState(false)
  const [regBusy, setRegBusy] = useState(false)

  const [logEmail, setLogEmail] = useState('')
  const [logPass, setLogPass] = useState('')
  const [logMsg, setLogMsg] = useState('')
  const [logErr, setLogErr] = useState(false)
  const [logBusy, setLogBusy] = useState(false)

  const handleRegister = async (e) => {
    e.preventDefault()
    setRegBusy(true)
    const fd = new FormData()
    fd.append('email', regEmail)
    fd.append('password', regPass)
    const res = await fetch('/register', { method: 'POST', body: fd })
    const body = await safeJson(res)
    setRegErr(!res.ok)
    setRegMsg(res.ok ? 'Registration successful. Redirecting...' : prettyError(body))
    setRegBusy(false)
    if (res.ok) onAuth()
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    setLogBusy(true)
    const fd = new FormData()
    fd.append('email', logEmail)
    fd.append('password', logPass)
    const res = await fetch('/login', { method: 'POST', body: fd })
    const body = await safeJson(res)
    setLogErr(!res.ok)
    setLogMsg(res.ok ? 'Login successful. Redirecting...' : prettyError(body))
    setLogBusy(false)
    if (res.ok) onAuth()
  }

  return (
    <>
      <section className="hero card">
        <div className="hero-topline">
          <p className="eyebrow">LangGraph Assignment</p>
          <span className="badge">FastAPI + Supabase + PGVector</span>
        </div>
        <h1>DocsQA Workspace</h1>
        <p className="lede">
          Upload PDFs, avoid duplicate reprocessing by file hash, and ask an agent that uses
          user-scoped document retrieval with optional web search.
        </p>
        <p className="developer-credit">Developed by Baba Kattubadi</p>
        {dbUnavailable && (
          <p className="db-warning">
            Database connection is temporarily unavailable. This is usually a transient DNS/network
            issue with the Supabase host. Please retry shortly.
          </p>
        )}
      </section>

      <section className="grid two auth-grid">
        <form className="card panel" onSubmit={handleRegister}>
          <div className="panel-head">
            <h2>Create account</h2>
            <p>Start by creating your personal docs workspace.</p>
          </div>
          <label>Email <input type="email" value={regEmail} onChange={e => setRegEmail(e.target.value)} required /></label>
          <label>Password <input type="password" value={regPass} onChange={e => setRegPass(e.target.value)} required /></label>
          <button type="submit" disabled={regBusy}>{regBusy ? 'Please wait...' : 'Register'}</button>
          {regMsg && <pre className={`result${regErr ? ' error' : ''}`}>{regMsg}</pre>}
        </form>

        <form className="card panel" onSubmit={handleLogin}>
          <div className="panel-head">
            <h2>Sign in</h2>
            <p>Continue with your existing account.</p>
          </div>
          <label>Email <input type="email" value={logEmail} onChange={e => setLogEmail(e.target.value)} required /></label>
          <label>Password <input type="password" value={logPass} onChange={e => setLogPass(e.target.value)} required /></label>
          <button type="submit" disabled={logBusy}>{logBusy ? 'Please wait...' : 'Login'}</button>
          {logMsg && <pre className={`result${logErr ? ' error' : ''}`}>{logMsg}</pre>}
        </form>
      </section>
    </>
  )
}

// ── main workspace ────────────────────────────────────────────────────────────

function Workspace({ user, documents: initialDocs, onLogout }) {
  const [docs, setDocs] = useState(initialDocs)
  const [activeTab, setActiveTab] = useState('documents')
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const [uploadErr, setUploadErr] = useState(false)
  const fileRef = useRef(null)

  const newSessionId = () => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  const [sessionId, setSessionId] = useState(() => {
    const stored = sessionStorage.getItem('chat_session_id')
    if (stored) return stored
    const id = newSessionId()
    sessionStorage.setItem('chat_session_id', id)
    return id
  })

  const storageKey = `chat_thread_${sessionId}`
  const defaultMsg = [{ role: 'assistant', text: 'Ask anything about your uploaded PDFs and I will answer with citations from retrieved chunks.', sources: undefined }]

  const [messages, setMessages] = useState(() => {
    try {
      const saved = sessionStorage.getItem(storageKey)
      return saved ? JSON.parse(saved) : defaultMsg
    } catch { return defaultMsg }
  })

  const [query, setQuery] = useState('')
  const [askBusy, setAskBusy] = useState(false)
  const threadRef = useRef(null)

  const saveThread = useCallback((msgs) => {
    sessionStorage.setItem(storageKey, JSON.stringify(msgs))
  }, [storageKey])

  useEffect(() => {
    if (threadRef.current)
      threadRef.current.scrollTop = threadRef.current.scrollHeight
  }, [messages])

  const handleNewChat = () => {
    const id = newSessionId()
    sessionStorage.setItem('chat_session_id', id)
    setSessionId(id)
    setMessages(defaultMsg)
    sessionStorage.setItem(`chat_thread_${id}`, JSON.stringify(defaultMsg))
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    const files = Array.from(fileRef.current?.files || [])
    if (!files.length) {
      setUploadErr(true); setUploadMsg('Please choose at least one PDF.'); return
    }
    setUploadBusy(true)
    const fd = new FormData()
    files.forEach(f => fd.append('file', f))
    const res = await fetch('/upload', { method: 'POST', body: fd })
    const body = await safeJson(res)
    if (res.ok) {
      const created = (body.documents || []).filter(d => d.created).length
      const reused = (body.documents || []).length - created
      setUploadErr(false)
      setUploadMsg(
        `Uploaded ${body.count} file(s). ${created} indexed, ${reused} reused.\n` +
        (body.documents || []).map(d => `- ${d.filename} (${d.page_count} pages)`).join('\n')
      )
      const updated = await fetch('/documents').then(r => r.json()).catch(() => docs)
      setDocs(updated)
      setTimeout(() => setUploadMsg(''), 4000)
    } else {
      setUploadErr(true)
      setUploadMsg(prettyError(body))
    }
    setUploadBusy(false)
  }

  const handleDelete = async (doc) => {
    if (!window.confirm(`Delete ${doc.filename} from your documents?`)) return
    const res = await fetch(`/documents/${doc.id}`, { method: 'DELETE' })
    if (res.ok) {
      setDocs(prev => prev.filter(d => d.id !== doc.id))
    } else {
      const body = await safeJson(res)
      setUploadErr(true); setUploadMsg(prettyError(body))
    }
  }

  const handleLogout = async (e) => {
    e.preventDefault()
    const res = await fetch('/logout', { method: 'POST' })
    if (res.ok) {
      sessionStorage.removeItem(storageKey)
      onLogout()
    }
  }

  const handleAsk = async (e) => {
    e.preventDefault()
    const q = query.trim()
    if (!q) return
    setQuery('')
    setAskBusy(true)

    const userMsg = { role: 'user', text: q }
    const pendingMsg = { role: 'assistant', text: 'Thinking...', pending: true }
    const next = [...messages, userMsg, pendingMsg]
    setMessages(next)
    saveThread(next)

    let answerText = ''
    let sources = null

    try {
      const res = await fetch('/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId },
        body: JSON.stringify({ query: q }),
      })

      if (!res.ok || !res.body) {
        const body = await safeJson(res)
        throw new Error(prettyError(body))
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const processEvent = (raw) => {
        const lines = raw.split('\n')
        let eventName = 'message'
        const dataLines = []
        for (const line of lines) {
          if (line.startsWith('event:')) eventName = line.slice(6).trim()
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
        }
        if (!dataLines.length) return
        const payload = JSON.parse(dataLines.join('\n'))

        if (eventName === 'token') {
          answerText += payload.content || ''
          setMessages(prev => {
            const updated = [...prev]
            updated[updated.length - 1] = { role: 'assistant', text: answerText, streaming: true }
            return updated
          })
        } else if (eventName === 'sources') {
          sources = payload.sources || null
        } else if (eventName === 'done') {
          answerText = payload.answer || answerText || 'Response received.'
          const final = [...messages, userMsg, { role: 'assistant', text: answerText, sources }]
          setMessages(final)
          saveThread(final)
        } else if (eventName === 'error') {
          throw new Error(payload.detail || 'Streaming failed.')
        }
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''
        for (const raw of events) { if (raw.trim()) processEvent(raw) }
      }
      buffer += decoder.decode()
      if (buffer.trim()) processEvent(buffer)

    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Request failed.'
      const errFinal = [...messages, userMsg, { role: 'assistant', text: errMsg, isError: true }]
      setMessages(errFinal)
      saveThread(errFinal)
    } finally {
      setAskBusy(false)
    }
  }

  return (
    <section className="app-layout">
      {/* Sidebar */}
      <aside className="card panel sidebar-panel">
        <div className="panel-head">
          <p className="eyebrow">LangGraph Assignment</p>
          <div className="sidebar-title-row">
            <div>
              <h1 className="sidebar-title">DocsQA Workspace</h1>
              <p className="muted">Private document chat with structured sources.</p>
              <p className="developer-credit">Developed by Baba Kattubadi</p>
            </div>
            <span className="badge">FastAPI + Supabase + PGVector</span>
          </div>
        </div>

        <div className="sidebar-tabs">
          <button type="button" className={`tab-btn${activeTab === 'documents' ? ' active' : ''}`} onClick={() => setActiveTab('documents')}>Documents</button>
          <button type="button" className={`tab-btn${activeTab === 'account' ? ' active' : ''}`} onClick={() => setActiveTab('account')}>Account</button>
        </div>

        {/* Documents tab */}
        <div className={`tab-panel${activeTab !== 'documents' ? ' hidden' : ''}`}>
          <form className="panel" onSubmit={handleUpload}>
            <label className="muted">Upload PDFs (max 5 files, 10 pages each)</label>
            <input type="file" ref={fileRef} accept="application/pdf" multiple required />
            <button type="submit" disabled={uploadBusy}>{uploadBusy ? 'Please wait...' : 'Upload'}</button>
          </form>
          {uploadMsg && <pre className={`result${uploadErr ? ' error' : ''}`}>{uploadMsg}</pre>}

          <div className="panel-head panel-head-inline">
            <h3>Your documents</h3>
            <span className="badge">{docs.length}</span>
          </div>
          <div className="docs sidebar-docs">
            {docs.length === 0
              ? <p className="muted">No documents uploaded yet.</p>
              : docs.map(doc => <DocCard key={doc.id} doc={doc} onDelete={handleDelete} />)
            }
          </div>
        </div>

        {/* Account tab */}
        <div className={`tab-panel${activeTab !== 'account' ? ' hidden' : ''}`}>
          <div className="panel-head account-head">
            <h2 className="user-email">{user.email}</h2>
            <p className="muted">Your uploaded docs are private to this account.</p>
          </div>
          <div id="logout-form-wrapper">
            <form onSubmit={handleLogout}>
              <button type="submit" className="secondary">Sign out</button>
            </form>
          </div>
        </div>
      </aside>

      {/* Chat */}
      <section className="card panel chat-shell chat-panel">
        <div className="panel-head panel-head-inline">
          <h2>DocsQA Chat</h2>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button type="button" className="secondary" style={{ fontSize: '0.875rem', padding: '6px 12px' }} onClick={handleNewChat}>
              New Chat
            </button>
            <span className="badge">Markdown enabled</span>
          </div>
        </div>

        <div className="chat-thread" ref={threadRef}>
          {messages.map((msg, i) =>
            msg.role === 'user'
              ? (
                <article className="chat-msg user" key={i}>
                  <div className="chat-bubble chat-bubble-user">{msg.text}</div>
                </article>
              ) : (
                <article className="chat-msg assistant" key={i}>
                  <AssistantBubble
                    text={msg.text}
                    sources={msg.sources}
                    pending={msg.pending || msg.streaming}
                    isError={msg.isError}
                  />
                </article>
              )
          )}
        </div>

        <form className="chat-composer" onSubmit={handleAsk}>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            rows={3}
            placeholder="Message DocsQA..."
            required
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.target.form.requestSubmit() } }}
          />
          <button type="submit" disabled={askBusy}>{askBusy ? 'Please wait...' : 'Send'}</button>
        </form>
      </section>
    </section>
  )
}

// ── root ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [appState, setAppState] = useState(null) // null = loading

  const fetchState = useCallback(async () => {
    try {
      const [docsRes, meRes] = await Promise.all([
        fetch('/documents'),
        fetch('/me'),
      ])
      if (docsRes.ok && meRes.ok) {
        const docs = await docsRes.json()
        const me = await meRes.json()
        setAppState({ loggedIn: true, user: { email: me.email }, documents: docs })
      } else {
        setAppState({ loggedIn: false })
      }
    } catch {
      setAppState({ loggedIn: false })
    }
  }, [])

  useEffect(() => { fetchState() }, [fetchState])

  if (appState === null) return null

  return (
    <>
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />
      <main className="shell">
        {appState.loggedIn
          ? <Workspace user={appState.user} documents={appState.documents} onLogout={() => setAppState({ loggedIn: false })} />
          : <AuthPage onAuth={fetchState} dbUnavailable={appState.dbUnavailable} />
        }
      </main>
    </>
  )
}
