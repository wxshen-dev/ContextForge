import { FormEvent, useEffect, useRef, useState } from 'react'
import { ArrowRight, BookOpen, Check, ChevronDown, CircleHelp, LoaderCircle, MessageSquareText, RotateCcw, Send, Sparkles, Trash2, Wifi, WifiOff } from 'lucide-react'
import { Link } from 'react-router'
import { apiJson, newSessionId, queryBase, type HistoryItem, type Progress } from '../lib/api'
import { ProgressSteps } from '../components/ProgressSteps'

type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
  imageUrls: string[]
  progress?: Progress
  pending?: boolean
}

const initialProgress: Progress = { status: 'processing', done_list: [], running_list: [] }
const suggestions = ['How do I measure DC voltage with the Atlas M42?', 'What should I check before measuring resistance?']

function getSessionId() {
  const stored = localStorage.getItem('kb_session_id')
  if (stored) return stored
  const id = newSessionId()
  localStorage.setItem('kb_session_id', id)
  return id
}

function cleanAnswer(raw: string): string {
  return raw.replace(/\n?\[Images\][\s\S]*$/i, '').trim()
}

function imageUrlsFromAnswer(raw: string): string[] {
  const block = raw.match(/\[Images\]([\s\S]*)$/i)?.[1] || ''
  return (block.match(/https?:\/\/[^\s<>"']+/g) || []).map((url) => url.replace(/[),.;]+$/, ''))
}

function readableError(cause: unknown): string {
  return cause instanceof Error ? cause.message : 'Something went wrong. Please try again.'
}

export function ChatPage() {
  const [sessionId] = useState(getSessionId)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [streaming, setStreaming] = useState(true)
  const [busy, setBusy] = useState(false)
  const [online, setOnline] = useState<boolean | null>(null)
  const [error, setError] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)
  const eventSource = useRef<EventSource | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    let active = true
    apiJson<{ items: HistoryItem[] }>(`${queryBase}/history/${encodeURIComponent(sessionId)}`)
      .then((data) => { if (active) setMessages(data.items.filter((item) => item.role === 'user' || item.role === 'assistant').map((item) => ({ id: item._id || crypto.randomUUID(), role: item.role as 'user' | 'assistant', text: cleanAnswer(item.text), imageUrls: item.image_urls?.length ? item.image_urls : imageUrlsFromAnswer(item.text) }))) })
      .catch((cause) => { if (active) setError(`Could not load conversation history: ${readableError(cause)}`) })
      .finally(() => { if (active) setLoadingHistory(false) })
    return () => { active = false; eventSource.current?.close() }
  }, [sessionId])

  useEffect(() => {
    let active = true
    async function check() {
      try {
        await apiJson<{ ok: boolean }>(`${queryBase}/health`)
        if (active) setOnline(true)
      } catch { if (active) setOnline(false) }
    }
    void check()
    const timer = window.setInterval(check, 10000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages])

  function updateAssistant(id: string, patch: Partial<Message>) {
    setMessages((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item))
  }

  async function send(text: string) {
    const value = text.trim()
    if (!value || busy) return
    const answerId = crypto.randomUUID()
    setError('')
    setBusy(true)
    setQuestion('')
    setMessages((current) => [...current,
      { id: crypto.randomUUID(), role: 'user', text: value, imageUrls: [] },
      { id: answerId, role: 'assistant', text: '', imageUrls: [], progress: initialProgress, pending: true },
    ])
    try {
      const result = await apiJson<{ session_id: string, answer?: string, image_urls?: string[], done_list?: string[] }>(`${queryBase}/query`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: value, session_id: sessionId, is_stream: streaming }),
      })
      if (!streaming) {
        updateAssistant(answerId, { text: cleanAnswer(result.answer || ''), imageUrls: result.image_urls?.length ? result.image_urls : imageUrlsFromAnswer(result.answer || ''), pending: false, progress: { status: 'completed', done_list: result.done_list || [], running_list: [] } })
        setBusy(false)
        return
      }
      const source = new EventSource(`${queryBase}/stream/${encodeURIComponent(result.session_id)}`)
      eventSource.current = source
      let answerText = ''
      const finish = () => { source.close(); if (eventSource.current === source) eventSource.current = null; setBusy(false) }
      source.addEventListener('progress', (event) => {
        const progress = JSON.parse((event as MessageEvent).data) as Progress
        updateAssistant(answerId, { progress })
      })
      source.addEventListener('delta', (event) => {
        const data = JSON.parse((event as MessageEvent).data) as { delta?: string }
        answerText += data.delta || ''
        updateAssistant(answerId, { text: cleanAnswer(answerText) })
      })
      source.addEventListener('final', (event) => {
        const data = JSON.parse((event as MessageEvent).data) as { answer?: string, image_urls?: string[] }
        const finalText = data.answer || answerText
        updateAssistant(answerId, { text: cleanAnswer(finalText), imageUrls: data.image_urls?.length ? data.image_urls : imageUrlsFromAnswer(finalText), pending: false, progress: { status: 'completed', done_list: [], running_list: [] } })
        void apiJson<Progress>(`${queryBase}/status/${encodeURIComponent(sessionId)}`).then((progress) => updateAssistant(answerId, { progress })).catch(() => {})
        finish()
      })
      source.addEventListener('error', (event) => {
        const detail = event instanceof MessageEvent ? (JSON.parse(event.data) as { error?: string }).error : 'Connection to the answer stream was lost.'
        setError(detail || 'The answer could not be completed.')
        updateAssistant(answerId, { pending: false, progress: { status: 'failed', done_list: [], running_list: [] } })
        finish()
      })
    } catch (cause) {
      setError(readableError(cause))
      updateAssistant(answerId, { pending: false, progress: { status: 'failed', done_list: [], running_list: [] } })
      setBusy(false)
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void send(question) }

  async function clearConversation() {
    if (busy || !window.confirm('Clear this conversation and its saved history?')) return
    try {
      await apiJson(`${queryBase}/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
      setMessages([])
      setError('')
      inputRef.current?.focus()
    } catch (cause) { setError(readableError(cause)) }
  }

  return <div className="chat-page">
    <aside className="chat-sidebar">
      <div className="sidebar-top"><p className="eyebrow">WORKSPACE / CHAT</p><h1>Ask your<br />knowledge base.</h1><p>Answers grounded in the documents you import.</p></div>
      <div className="sidebar-section"><div className="sidebar-label">YOUR WORKSPACE</div><div className="sidebar-item selected"><MessageSquareText size={17} />Current conversation</div><Link to="/import" className="sidebar-item"><BookOpen size={17} />Import documents <ArrowRight size={15} className="sidebar-arrow" /></Link></div>
      <div className="sidebar-bottom"><div className="sidebar-tip"><CircleHelp size={18} /><span>Ask about a product, procedure, or detail from your documents.</span></div><div className="sidebar-footer">CONTEXTFORGE / KNOWLEDGE ASSISTANT</div></div>
    </aside>
    <section className="conversation" aria-label="Conversation">
      <header className="conversation-header"><div><p className="eyebrow">CONVERSATION</p><h2>Knowledge assistant</h2></div><div className="conversation-actions"><span className={`connection-status ${online === false ? 'offline' : ''}`}>{online === false ? <WifiOff size={14} /> : <Wifi size={14} />}{online === false ? 'Service offline' : online === null ? 'Connecting' : 'Connected'}</span><button className="icon-button" type="button" onClick={clearConversation} disabled={busy || !messages.length} aria-label="Clear conversation" title="Clear conversation"><Trash2 size={18} /></button></div></header>
      <div className="message-scroll">
        {loadingHistory ? <div className="chat-loading"><LoaderCircle className="spin" size={23} /> Loading conversation…</div> : messages.length === 0 ? <div className="chat-welcome"><span className="welcome-mark"><Sparkles size={27} /></span><p className="eyebrow">START A CONVERSATION</p><h2>What would you like to know?</h2><p>Ask a question about your imported documents and follow each step of the answer.</p><div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} type="button" onClick={() => void send(suggestion)}>{suggestion}<ArrowRight size={16} /></button>)}</div></div> : <div className="messages">{messages.map((message) => <div className={`message-row ${message.role}`} key={message.id}><div className={`message-avatar ${message.role}`} aria-hidden="true">{message.role === 'assistant' ? <Sparkles size={18} /> : <span>You</span>}</div><div className="message-body"><div className="message-meta">{message.role === 'assistant' ? 'ContextForge' : 'You'}</div>{message.role === 'user' ? <div className="user-message">{message.text}</div> : <><div className="assistant-answer">{message.text ? <p>{message.text}</p> : <span className="thinking"><LoaderCircle size={16} className="spin" /> Finding an answer…</span>}{message.imageUrls.length > 0 && <div className="answer-images">{message.imageUrls.map((url) => <a href={url} target="_blank" rel="noreferrer" key={url}><img src={url} alt="Document reference" loading="lazy" /></a>)}</div>}</div>{message.progress && (message.progress.done_list.length > 0 || message.progress.running_list.length > 0 || message.pending) && <details className="answer-progress"><summary><span className="progress-summary-icon">{message.pending ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}</span><span>{message.pending ? 'Working through your question' : `${message.progress.done_list.length} steps completed`}</span><ChevronDown size={15} className="summary-chevron" /></summary><ProgressSteps progress={message.progress} compact /></details>}</>}</div></div>)}<div ref={endRef} /></div>}
      </div>
      <div className="composer-area">{error && <p className="chat-error" role="alert">{error}<button type="button" onClick={() => setError('')} aria-label="Dismiss error"><RotateCcw size={15} /></button></p>}<form className="composer" onSubmit={onSubmit}><label className="visually-hidden" htmlFor="question">Ask a question</label><textarea ref={inputRef} id="question" placeholder="Ask a question about your documents…" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send(question) } }} rows={2} disabled={busy} /><div className="composer-tools"><label className="stream-toggle"><input type="checkbox" checked={streaming} onChange={(event) => setStreaming(event.target.checked)} disabled={busy} /><span className="toggle-track" />Stream answer</label><button className="send-button" type="submit" disabled={busy || !question.trim()} aria-label="Send question">{busy ? <LoaderCircle size={19} className="spin" /> : <Send size={19} />}</button></div></form><p className="composer-caption">Answers are generated from your knowledge base. Verify important details in the source document.</p></div>
    </section>
  </div>
}
