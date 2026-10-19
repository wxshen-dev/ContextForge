export type Progress = {
  status: string
  done_list: string[]
  running_list: string[]
}

export type ImportTask = Progress & {
  task_id: string
}

export type HistoryItem = {
  _id: string
  role: string
  text: string
  image_urls?: string[]
  ts?: string
}

export const queryBase = import.meta.env.VITE_QUERY_API_URL || (import.meta.env.DEV ? '/api/query' : '')
export const importBase = import.meta.env.VITE_IMPORT_API_URL || (import.meta.env.DEV
  ? '/api/import'
  : `${window.location.protocol}//${window.location.hostname}:8000`)

export async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const data = await response.json() as { detail?: string }
      if (typeof data.detail === 'string') detail = data.detail
    } catch { /* Keep the HTTP status when the response is not JSON. */ }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export function newSessionId(): string {
  return `sess-${crypto.randomUUID()}`
}
