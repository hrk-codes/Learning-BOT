export type Workspace = { id: string; name: string; slug: string; created_at: string }
export type User = { id: string; email: string; display_name: string; role: string; created_at: string }
export type Session = { access_token: string; token_type: string; expires_in: number; user: User; workspace: Workspace }

export type RunEvent = {
  id: number
  event_type: string
  message: string
  node: string | null
  status: string | null
  public_data: Record<string, unknown>
  created_at: string
}

export type Run = {
  id: string
  workspace_id: string
  goal: string
  mode: string
  status: string
  progress: number
  current_node: string | null
  result_text: string | null
  error_code: string | null
  error_message: string | null
  trace_id: string
  llm_calls: number
  input_tokens: number
  output_tokens: number
  tool_calls: number
  estimated_cost_usd: number
  queue_seconds: number
  execution_seconds: number
  created_at: string
  started_at: string | null
  completed_at: string | null
  events?: RunEvent[]
}

export type Tool = {
  name: string
  description: string
  permission: string
  side_effect: string
  requires_confirmation: boolean
  enabled: boolean
  input_schema: Record<string, unknown>
}

export type MemoryRecord = {
  id: string
  workspace_id: string | null
  memory_type: string
  content: string
  is_active: boolean
  created_at: string
  updated_at: string
}

export type Document = {
  id: string
  workspace_id: string
  filename: string
  content_type: string
  size_bytes: number
  status: string
  chunk_count: number
  version: string
  created_at: string
  indexed_at: string | null
}

export type Approval = {
  id: string
  run_id: string
  action_type: string
  summary: string
  proposal: Record<string, unknown>
  status: string
  created_at: string
  decided_at: string | null
}

export type Analytics = {
  total_runs: number
  completed_runs: number
  failed_runs: number
  active_runs: number
  approval_rate: number
  average_execution_seconds: number
  p95_execution_seconds: number
  total_llm_calls: number
  total_tokens: number
  total_tool_calls: number
  status_counts: Record<string, number>
  daily_runs: { date: string; runs: number }[]
}
