import type { Analytics, Approval, Document, MemoryRecord, Run, Session, Tool } from './types'

const API = '/api/v1'

type ApiErrorBody = { error?: { message?: string; request_id?: string } }

async function request<T>(path: string, token?: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (!(init.body instanceof FormData) && init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${API}${path}`, { ...init, headers })
  if (!response.ok) {
    let body: ApiErrorBody = {}
    try { body = await response.json() as ApiErrorBody } catch { /* response was not JSON */ }
    const requestId = body.error?.request_id ? ` (${body.error.request_id})` : ''
    throw new Error(`${body.error?.message ?? `Request failed with ${response.status}`}${requestId}`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  register: (email: string, password: string, displayName: string) => request<Session>('/auth/register', undefined, { method: 'POST', body: JSON.stringify({ email, password, display_name: displayName }) }),
  login: (email: string, password: string) => {
    const body = new URLSearchParams({ username: email, password })
    return request<Session>('/auth/token', undefined, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body })
  },
  runs: (token: string, workspace: string) => request<Run[]>(`/runs?workspace_id=${encodeURIComponent(workspace)}`, token),
  run: (token: string, id: string) => request<Run>(`/runs/${id}`, token),
  createRun: (token: string, workspace_id: string, goal: string, mode: string) => request<Run>('/runs', token, { method: 'POST', headers: { 'Idempotency-Key': crypto.randomUUID() }, body: JSON.stringify({ workspace_id, goal, mode }) }),
  cancelRun: (token: string, id: string) => request<Run>(`/runs/${id}/cancel`, token, { method: 'POST' }),
  tools: (token: string, workspace: string) => request<Tool[]>(`/tools?workspace_id=${encodeURIComponent(workspace)}`, token),
  updateTool: (token: string, workspace: string, name: string, enabled: boolean) => request<Tool>(`/tools/${encodeURIComponent(name)}?workspace_id=${encodeURIComponent(workspace)}`, token, { method: 'PATCH', body: JSON.stringify({ enabled }) }),
  memories: (token: string, workspace: string) => request<MemoryRecord[]>(`/memories?workspace_id=${encodeURIComponent(workspace)}`, token),
  createMemory: (token: string, workspace_id: string, memory_type: string, content: string) => request<MemoryRecord>('/memories', token, { method: 'POST', body: JSON.stringify({ workspace_id, memory_type, content }) }),
  deleteMemory: (token: string, id: string) => request<void>(`/memories/${id}`, token, { method: 'DELETE' }),
  documents: (token: string, workspace: string) => request<Document[]>(`/documents?workspace_id=${encodeURIComponent(workspace)}`, token),
  uploadDocument: (token: string, workspace: string, file: File) => {
    const body = new FormData()
    body.set('workspace_id', workspace)
    body.set('version', '1')
    body.set('file', file)
    return request<Document>('/documents', token, { method: 'POST', body })
  },
  deleteDocument: (token: string, id: string) => request<void>(`/documents/${id}`, token, { method: 'DELETE' }),
  approvals: (token: string, workspace: string) => request<Approval[]>(`/approvals?workspace_id=${encodeURIComponent(workspace)}`, token),
  decideApproval: (token: string, id: string, decision: 'approved' | 'rejected') => request<Approval>(`/approvals/${id}/decision`, token, { method: 'POST', body: JSON.stringify({ decision }) }),
  analytics: (token: string, workspace: string) => request<Analytics>(`/analytics?workspace_id=${encodeURIComponent(workspace)}`, token),
}

export async function streamRunEvents(token: string, runId: string, onEvent: () => void, signal: AbortSignal): Promise<void> {
  const response = await fetch(`${API}/runs/${runId}/events`, { headers: { Authorization: `Bearer ${token}` }, signal })
  if (!response.ok || !response.body) throw new Error('Could not open the run event stream.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      if (frame.includes('event: run_event') || frame.includes('event: done')) onEvent()
    }
  }
}
