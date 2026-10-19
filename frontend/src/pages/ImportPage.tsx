import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react'
import { ArrowUpRight, Check, FileText, LoaderCircle, UploadCloud, X } from 'lucide-react'
import { Link } from 'react-router'
import { apiJson, importBase, type ImportTask } from '../lib/api'
import { ProgressSteps } from '../components/ProgressSteps'

type FileTask = ImportTask & { name: string }

function savedTasks(): FileTask[] {
  try {
    const value = JSON.parse(localStorage.getItem('cf_import_tasks') || '[]') as unknown
    return Array.isArray(value) ? value.filter((item): item is FileTask =>
      typeof item === 'object' && item !== null && typeof item.task_id === 'string' && typeof item.name === 'string',
    ).slice(0, 20) : []
  } catch { return [] }
}

export function ImportPage() {
  const [files, setFiles] = useState<File[]>([])
  const [tasks, setTasks] = useState<FileTask[]>(savedTasks)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { localStorage.setItem('cf_import_tasks', JSON.stringify(tasks.slice(0, 20))) }, [tasks])

  useEffect(() => {
    if (!tasks.some((task) => !['completed', 'failed'].includes(task.status))) return
    const timer = window.setInterval(async () => {
      for (const task of tasks.filter((item) => !['completed', 'failed'].includes(item.status))) {
        try {
          const next = await apiJson<ImportTask>(`${importBase}/status/${encodeURIComponent(task.task_id)}`)
          setTasks((current) => current.map((item) => item.task_id === next.task_id ? { ...item, ...next } : item))
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : 'Could not check import progress.')
        }
      }
    }, 2000)
    return () => window.clearInterval(timer)
  }, [tasks])

  function addFiles(incoming: FileList | File[]) {
    const valid = Array.from(incoming).filter((file) => /\.(pdf|md)$/i.test(file.name))
    if (valid.length !== incoming.length) setError('Choose PDF or Markdown files only.')
    else setError('')
    setFiles((current) => [...current, ...valid])
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    addFiles(event.dataTransfer.files)
  }

  function onPick(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) addFiles(event.target.files)
    event.target.value = ''
  }

  async function upload() {
    if (!files.length || uploading) return
    setUploading(true)
    setError('')
    const batch = files
    const form = new FormData()
    batch.forEach((file) => form.append('files', file))
    try {
      const result = await apiJson<{ task_ids: string[] }>(`${importBase}/upload`, { method: 'POST', body: form })
      setTasks((current) => [
        ...result.task_ids.map((task_id, index) => ({ task_id, name: batch[index].name, status: 'processing', done_list: [], running_list: [] })),
        ...current,
      ])
      setFiles([])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const completed = tasks.filter((task) => task.status === 'completed').length
  return <div className="page-wrap import-page">
    <section className="page-intro">
      <div><p className="eyebrow">KNOWLEDGE BASE / IMPORT</p><h1>Make your documents searchable.</h1><p className="lead">Add PDF or Markdown files. ContextForge processes them into passages ready for grounded answers.</p></div>
      <Link to="/chat" className="text-link">Open chat <ArrowUpRight size={17} aria-hidden="true" /></Link>
    </section>
    <div className="workspace-grid">
      <section className="surface upload-panel" aria-labelledby="upload-title">
        <div className="panel-heading"><div><p className="section-kicker">01 / SOURCE FILES</p><h2 id="upload-title">Upload documents</h2></div><span className="subtle-tag">PDF &amp; Markdown</span></div>
        <div className={dragging ? 'drop-zone dragging' : 'drop-zone'} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
          <span className="drop-icon"><UploadCloud size={29} strokeWidth={1.7} /></span>
          <h3>Drop files here</h3><p>Or choose files from your computer.</p>
          <button className="secondary-button" type="button" onClick={() => inputRef.current?.click()}>Browse files</button>
          <input ref={inputRef} className="visually-hidden" type="file" accept=".pdf,.md,application/pdf,text/markdown" multiple onChange={onPick} aria-label="Choose PDF or Markdown files" />
        </div>
        {files.length > 0 && <div className="selected-files"><div className="small-heading">READY TO UPLOAD <span>{files.length}</span></div>{files.map((file, index) => <div className="selected-file" key={`${file.name}-${index}`}><FileText size={17} /><span>{file.name}</span><button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles((current) => current.filter((_, position) => position !== index))}><X size={16} /></button></div>)}</div>}
        {error && <p className="inline-error" role="alert">{error}</p>}
        <button className="primary-button upload-action" type="button" disabled={!files.length || uploading} onClick={upload}>{uploading ? <LoaderCircle className="spin" size={17} /> : <UploadCloud size={17} />} {uploading ? 'Uploading…' : `Upload ${files.length || ''} ${files.length === 1 ? 'file' : 'files'}`}</button>
      </section>
      <section className="surface activity-panel" aria-labelledby="activity-title">
        <div className="panel-heading"><div><p className="section-kicker">02 / PROCESSING</p><h2 id="activity-title">Import activity</h2></div><span className="count-pill">{completed} completed</span></div>
        {tasks.length === 0 ? <div className="activity-empty"><span className="empty-icon"><FileText size={26} /></span><h3>No imports yet</h3><p>Upload a document to see its processing stages here.</p></div> : <div className="task-list">{tasks.map((task) => <article className="task-card" key={task.task_id}><div className="task-header"><span className="task-file-icon"><FileText size={18} /></span><div className="task-title"><strong>{task.name}</strong><small>{task.status === 'completed' ? 'Ready to search' : task.status === 'failed' ? 'Import failed' : 'Processing document'}</small></div><span className={`status-badge ${task.status}`}>{task.status === 'completed' ? <Check size={13} /> : task.status === 'failed' ? <X size={13} /> : <LoaderCircle size={13} className="spin" />}{task.status}</span></div><ProgressSteps progress={task} compact /></article>)}</div>}
      </section>
    </div>
  </div>
}
