export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  rewritten_query?: string
  item_names?: string[]
  image_urls?: string[]
  timestamp: number
  sources?: ChunkSource[]
  node_steps?: any[]
  total_duration?: number
  isStreaming?: boolean
  isStepsCollapsed?: boolean
}

export interface ChunkSource {
  chunk_id: string | number
  file_title?: string
  parent_title?: string
  item_name?: string
  title?: string
  content: string
  score?: number
  source?: 'local' | 'web'
  url?: string
}

export interface ChatSession {
  session_id: string
  title: string
  last_message: string
  last_role: string
  last_ts: number
  message_count: number
  item_names: string[]
  messages?: ChatMessage[]
}

export interface CandidateItem {
  id: string
  item_name: string
  file_title?: string
  score?: number
  description?: string
}

export interface KBItem {
  id: string
  item_name: string
  file_title: string
  chunk_count: number
  created_at: string
  file_size?: string
  dense_dim: number
  has_sparse: boolean
}

export interface KBChunk {
  chunk_id: number | string
  file_title: string
  item_name: string
  title: string
  parent_title: string
  part: number
  content: string
  dense_vector_preview?: number[]
  sparse_vector_preview?: Record<number, number>
}

export interface PipelineNodeStatus {
  node_id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  updated_at?: string
  error_msg?: string
  duration?: number
}

export interface ImportTask {
  task_id: string
  file_name: string
  file_size: string
  status: 'processing' | 'completed' | 'failed' | 'waiting'
  created_at: string
  error_msg?: string
  raw_error?: string
  total_duration?: number
  nodes: PipelineNodeStatus[]
}

export interface SystemStats {
  total_items: number
  total_chunks: number
  total_sessions: number
  milvus_status: 'online' | 'offline' | 'degraded'
  minio_status: 'online' | 'offline' | 'degraded'
  mongo_status: 'online' | 'offline' | 'degraded'
}
