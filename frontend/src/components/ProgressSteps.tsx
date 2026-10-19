import { Check, Circle, LoaderCircle } from 'lucide-react'
import type { Progress } from '../lib/api'

const labels: Record<string, string> = {
  upload_file: 'Upload file',
  entry: 'Check file',
  pdf_to_md: 'Convert PDF to Markdown',
  md_img: 'Process document images',
  document_split: 'Split document',
  item_name_recognition: 'Identify item name',
  bge_embedding: 'Generate vectors',
  import_milvus: 'Import vector store',
  item_name_confirm: 'Identify the product',
  web_search_mcp: 'Check web sources',
  search_embedding: 'Search document passages',
  search_embedding_hyde: 'Expand the search',
  rrf: 'Combine search results',
  rerank: 'Rank matching passages',
  answer_output: 'Write the answer',
}

function labelFor(step: string): string {
  const normalized = step.replace(/^node_/, '')
  const fallback = normalized.replaceAll('_', ' ')
  return labels[normalized] || fallback.charAt(0).toUpperCase() + fallback.slice(1)
}

export function ProgressSteps({ progress, compact = false }: { progress: Progress, compact?: boolean }) {
  const steps = [...new Set([...progress.done_list, ...progress.running_list])]
  if (!steps.length) return <p className="progress-empty">Waiting for the first step…</p>
  return (
    <ol className={compact ? 'step-list compact' : 'step-list'}>
      {steps.map((step) => {
        const done = progress.done_list.includes(step)
        return <li key={step} className={done ? 'step done' : 'step running'}>
          <span className="step-icon" aria-hidden="true">{done ? <Check size={13} /> : progress.running_list.includes(step) ? <LoaderCircle size={13} className="spin" /> : <Circle size={13} />}</span>
          <span>{labelFor(step)}</span>
        </li>
      })}
    </ol>
  )
}
