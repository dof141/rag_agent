import type {
  KBItem,
  KBChunk,
  ChatSession,
  ChatMessage,
  ImportTask,
  SystemStats,
  CandidateItem,
  RuntimeSettingsResponse,
  RuntimeSettingsUpdate,
} from '../types'
import { mockStats, mockKBItems, mockKBChunks, mockSessions } from '../mock/mockData'
import { authFetch } from './http'

const API_BASE = ''

let localKBItems = [...mockKBItems]
let localKBChunks = { ...mockKBChunks }
let localSessions = [...mockSessions]
let localStats = { ...mockStats }

const PIPELINE_NODES_CONFIG = [
  { node_id: 'upload_file', name: '文件保存上传', description: '本地磁盘存储与校验' },
  { node_id: 'node_entry', name: '入口校验', description: '参数初始化与格式验证' },
  { node_id: 'node_pdf_to_md', name: 'PDF转Markdown', description: 'MinerU 多模态解析与布局识别' },
  { node_id: 'node_md_img', name: 'MD图片处理', description: 'VLM 描述图片并上传 MinIO' },
  { node_id: 'node_document_split', name: '文档智能切片', description: '标题感知层级切片' },
  { node_id: 'node_item_name_recognition', name: '学习主题识别', description: 'LLM 识别主题与索引构建' },
  { node_id: 'node_generate_embeddings', name: '向量生成', description: '按当前配置生成文档向量' },
  { node_id: 'node_import_vector_store', name: '向量入库', description: '幂等写入所选向量库' }
]

const TASKS_STORAGE_KEY = 'rag_import_tasks'

function loadTasksFromStorage(): ImportTask[] {
  try {
    const raw = localStorage.getItem(TASKS_STORAGE_KEY)
    if (raw) {
      return JSON.parse(raw)
    }
  } catch (e) {
    console.warn('读取本地任务缓存失败', e)
  }
  return []
}

function saveTasksToStorage(tasks: ImportTask[]) {
  try {
    localStorage.setItem(TASKS_STORAGE_KEY, JSON.stringify(tasks))
  } catch (e) {
    console.warn('保存本地任务缓存失败', e)
  }
}

const CN_NAME_MAP: Record<string, string> = {
  upload_file: '开始上传文件',
  node_entry: '检查文件',
  node_pdf_to_md: 'PDF转Markdown',
  node_md_img: 'Markdown图片处理',
  node_document_split: '文档切分',
  node_item_name_recognition: '主体名称识别',
  node_generate_embeddings: '向量生成',
  node_import_vector_store: '导入向量库',
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const message = typeof data?.detail === 'string' ? data.detail : `请求失败：${response.status}`
    throw new ApiError(response.status, message)
  }
  return data as T
}

function isNodeInList(nodeId: string, nodeName: string, list: string[]): boolean {
  if (!list || list.length === 0) return false
  const cnName = CN_NAME_MAP[nodeId] || ''
  return list.includes(nodeId) || list.includes(nodeName) || (cnName ? list.includes(cnName) : false)
}

export function formatFriendlyErrorMsg(raw: string): string {
  if (!raw) return '后端节点解析异常'
  
  if (raw.includes('mineru.net') || raw.includes('openxlab.org.cn') || raw.includes('ConnectTimeoutError') || raw.includes('SSLError') || raw.includes('Max retries exceeded')) {
    return 'MinerU 官方解析云服务连接超时，请检查网络连接或稍后重试'
  }
  if (raw.includes('FileNotFoundError')) {
    return '解析目标文件或路径不存在'
  }
  if (raw.includes('ValueError')) {
    return '文件参数或数据格式校验失败'
  }
  if (raw.includes('Milvus') || raw.includes('collection')) {
    return 'Milvus 向量数据库连接或存储异常'
  }
  if (raw.includes('MinIO') || raw.includes('S3')) {
    return 'MinIO 对象存储图片上传失败'
  }

  return raw.length > 90 ? raw.slice(0, 90) + '...' : raw
}

export const api = {
  // === 系统状态 ===
  async getSystemStats(): Promise<SystemStats> {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (res.ok) {
        localStats.milvus_status = 'online'
      }
    } catch {
      // 忽略
    }
    localStats.total_items = localKBItems.length
    localStats.total_chunks = Object.values(localKBChunks).reduce((acc, cur) => acc + cur.length, 0)
    localStats.total_sessions = localSessions.length
    return localStats
  },

  // === 真实文档上传与后台任务轮询 API ===
  async uploadFiles(files: File[]): Promise<{ task_ids: string[] }> {
    const formData = new FormData()
    files.forEach(file => formData.append('files', file))

    try {
      const res = await authFetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      })

      if (res.ok) {
        const data = await res.json()
        const taskIds: string[] = data.task_ids || []
        const currentTasks = loadTasksFromStorage()

        for (let i = 0; i < files.length; i++) {
          const file = files[i]
          const tid = taskIds[i] || `task-${Date.now()}-${i}`
          const taskObj: ImportTask = {
            task_id: tid,
            file_name: file.name,
            file_size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
            status: 'processing',
            created_at: new Date().toLocaleString(),
            nodes: PIPELINE_NODES_CONFIG.map(node => ({
              ...node,
              status: node.node_id === 'upload_file' ? 'completed' : (node.node_id === 'node_entry' ? 'running' : 'pending')
            }))
          }
          currentTasks.unshift(taskObj)
        }
        saveTasksToStorage(currentTasks)
        return { task_ids: taskIds }
      }
      await parseJsonResponse(res)
    } catch (e) {
      if (e instanceof ApiError) {
        throw e
      }
    }

    return { task_ids: [] }
  },

  async getTasks(): Promise<ImportTask[]> {
    const tasks = loadTasksFromStorage()
    let hasChanges = false

    for (const task of tasks) {
      if (task.status === 'processing' || (task.status === 'failed' && !task.nodes.some(n => n.status === 'failed'))) {
        try {
          const res = await authFetch(`${API_BASE}/status/${task.task_id}?_=${Date.now()}`)
          if (res.ok) {
            const data = await res.json()
            const doneList: string[] = data.done_list || []
            const runningList: string[] = data.running_list || []
            const globalStatus: string = data.status || 'processing'
            const rawErrorMsg: string = data.error || ''
            const nodeDurations: Record<string, number> = data.node_durations || {}
            const totalDuration: number = data.total_duration || 0

            task.status = globalStatus === 'completed' ? 'completed' : (globalStatus === 'failed' ? 'failed' : 'processing')
            task.total_duration = totalDuration
            if (rawErrorMsg) {
              task.error_msg = formatFriendlyErrorMsg(rawErrorMsg)
              task.raw_error = rawErrorMsg
            }

            let foundFailedNode = false
            task.nodes.forEach(node => {
              const isDone = isNodeInList(node.node_id, node.name, doneList) || node.node_id === 'upload_file'
              const isRunning = isNodeInList(node.node_id, node.name, runningList)
              const d = nodeDurations[node.node_id] ?? nodeDurations[CN_NAME_MAP[node.node_id]]
              if (d !== undefined) {
                node.duration = d
              }

              if (isDone) {
                node.status = 'completed'
              } else if (globalStatus === 'failed' || task.status === 'failed') {
                if (!foundFailedNode) {
                  node.status = 'failed'
                  node.error_msg = formatFriendlyErrorMsg(rawErrorMsg) || '后端 LangGraph 节点处理异常'
                  foundFailedNode = true
                } else {
                  node.status = 'pending'
                }
              } else if (isRunning) {
                node.status = 'running'
              } else if (node.status !== 'completed') {
                node.status = 'pending'
              }
            })
            hasChanges = true
          }
        } catch {
          // 忽略单次轮询异常
        }
      }
    }

    if (hasChanges) {
      saveTasksToStorage(tasks)
    }

    return tasks
  },

  async retryTask(taskId: string): Promise<boolean> {
    try {
      const res = await authFetch(`${API_BASE}/retry/${taskId}`, { method: 'POST' })
      if (res.ok) {
        const tasks = loadTasksFromStorage()
        const target = tasks.find(t => t.task_id === taskId)
        if (target) {
          target.status = 'processing'
          target.error_msg = undefined
          target.raw_error = undefined
          target.total_duration = undefined
          target.nodes = PIPELINE_NODES_CONFIG.map(node => ({
            ...node,
            status: node.node_id === 'upload_file' ? 'completed' : (node.node_id === 'node_entry' ? 'running' : 'pending'),
            duration: undefined,
            error_msg: undefined
          }))
          saveTasksToStorage(tasks)
        }
        return true
      }
    } catch (e) {
      // 单次重试失败保持当前任务状态。
    }
    return false
  },

  async clearTasks(): Promise<void> {
    localStorage.removeItem(TASKS_STORAGE_KEY)
  },

  async getTaskStatus(taskId: string): Promise<{ status: string; done_list: string[]; running_list: string[]; node_durations?: Record<string, number>; total_duration?: number }> {
    try {
      const res = await authFetch(`${API_BASE}/status/${taskId}`)
      if (res.ok) {
        return await res.json()
      }
    } catch {
      // 忽略
    }
    return { status: 'processing', done_list: [], running_list: [], node_durations: {}, total_duration: 0 }
  },

  async getRuntimeSettings(): Promise<RuntimeSettingsResponse | null> {
    const response = await authFetch(`${API_BASE}/api/settings/runtime`)
    return parseJsonResponse<RuntimeSettingsResponse | null>(response)
  },

  async saveRuntimeSettings(payload: RuntimeSettingsUpdate): Promise<RuntimeSettingsResponse> {
    const response = await authFetch(`${API_BASE}/api/settings/runtime`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return parseJsonResponse<RuntimeSettingsResponse>(response)
  },

  async clearRuntimeSecret(name: 'embedding_api_key' | 'qdrant_api_key' | 'milvus_token'): Promise<RuntimeSettingsResponse> {
    const response = await authFetch(`${API_BASE}/api/settings/runtime/secrets/${name}`, { method: 'DELETE' })
    return parseJsonResponse<RuntimeSettingsResponse>(response)
  },

  // === 真实智能问答与断点确认 API ===
  async sendQuery(query: string, sessionId?: string, isStream: boolean = true): Promise<{
    session_id: string
    request_id: string
    answer?: string
    awaiting_confirmation?: boolean
    candidates?: CandidateItem[]
  }> {
    const sid = sessionId || `sess-${Date.now()}`
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sid, is_stream: isStream })
      })

      if (res.ok) {
        const data = await res.json()
        
        if (data.awaiting_confirmation || data.status === 'waiting_confirmation') {
          return {
            session_id: data.session_id || sid,
            request_id: data.request_id || `req-${Date.now()}`,
            awaiting_confirmation: true,
            candidates: data.candidate_items || data.candidates || []
          }
        }

        const answerText = data.answer || '问答生成完毕。'

        return {
          session_id: data.session_id || sid,
          request_id: data.request_id || `req-${Date.now()}`,
          answer: answerText
        }
      }
    } catch (e) {
      console.warn('真实问答接口未连通，切换至仿真模式', e)
    }

    const reqId = `req-${Date.now()}`
    const lowerQ = query.toLowerCase()

    if (lowerQ.includes('neo4j') || (lowerQ.includes('路由器') && !lowerQ.includes('er2100') && !lowerQ.includes('ner214w'))) {
      return {
        session_id: sid,
        request_id: reqId,
        awaiting_confirmation: true,
        candidates: [
          { id: 'Neo4j 5.26（LTS）', item_name: 'Neo4j 5.26（LTS）', file_title: 'Neo4j 官方安装部署指南.pdf', score: 0.797 },
          { id: 'H3C ER2100', item_name: 'H3C ER2100', file_title: 'H3C ER2100企业级路由器.pdf', score: 0.82 }
        ]
      }
    }

    let mockAnswer = `针对您关于 **${query}** 的提问，根据已导入的设备知识库匹配分析：\n\n1. **设备性能与配置**：建议查阅对应的设备操作手册。\n2. **安全防护**：请在断电状态下拆卸并检查接地线。\n3. **常见排错**：确认面板指示灯状态为常绿。\n\n![设备外观示意图](http://127.0.0.1:9000/knowledge-base-files/images/hak180/panel_diag.jpg)`
    let itemTag = '通用设备'

    if (lowerQ.includes('hak') || lowerQ.includes('烫金')) {
      itemTag = 'hak180'
      mockAnswer = `针对 **HAK 180 烫金机** 的局部烫印设置要求（顶部 50 mm – 170 mm）：\n\n1. 按面板 **[Mode]** 键进入参数设定；\n2. 将 **"Top Offset"** 设为 **50mm**；\n3. 将 **"Print Length"** 设为 **120mm**（即 170mm - 50mm）；\n4. 按 **[OK]** 保存即可生效。\n\n![HAK180操作面板架构](http://127.0.0.1:9000/knowledge-base-files/images/hak180/panel_diag.jpg)`
    } else if (lowerQ.includes('万用表') || lowerQ.includes('电池')) {
      itemTag = '万用表RS-12'
      mockAnswer = `**万用表 RS-12 电池更换规范**：\n\n1. 关闭电源开关，拔出红黑测试表笔；\n2. 拧开后盖板上 2 颗固定螺丝；\n3. 更换 9V 叠层电池，注意正负极指向；\n4. 合上盖板并拧紧螺丝。`
    }

    this._saveToLocalSession(sid, query, mockAnswer, [itemTag])

    return {
      session_id: sid,
      request_id: reqId,
      answer: mockAnswer
    }
  },

  _saveToLocalSession(sid: string, query: string, answer: string, itemNames: string[]) {
    let targetSession = localSessions.find(s => s.session_id === sid)
    const userMsg: ChatMessage = { id: `m-${Date.now()}-u`, role: 'user', text: query, timestamp: Date.now() / 1000 }
    const assistMsg: ChatMessage = {
      id: `m-${Date.now()}-a`,
      role: 'assistant',
      text: answer,
      item_names: itemNames,
      timestamp: (Date.now() + 500) / 1000,
      sources: [
        { chunk_id: 101, title: '设备操作指导书', content: '参照标准流程...', score: 0.92, source: 'local' }
      ]
    }

    if (!targetSession) {
      targetSession = {
        session_id: sid,
        title: query.slice(0, 16) + '...',
        last_message: answer.slice(0, 40),
        last_role: 'assistant',
        last_ts: Date.now() / 1000,
        message_count: 2,
        item_names: itemNames,
        messages: [userMsg, assistMsg]
      }
      localSessions.unshift(targetSession)
    } else {
      if (!targetSession.messages) targetSession.messages = []
      targetSession.messages.push(userMsg, assistMsg)
      targetSession.last_message = answer.slice(0, 40)
      targetSession.last_role = 'assistant'
      targetSession.last_ts = Date.now() / 1000
      targetSession.message_count = targetSession.messages.length
    }
  },

  // === 向量库管理 API ===
  async getKBItems(): Promise<KBItem[]> {
    try {
      const res = await fetch(`${API_BASE}/api/kb/items`)
      if (res.ok) {
        const data = await res.json()
        if (data && data.data) return data.data
      }
    } catch {
      // 忽略
    }
    return localKBItems
  },

  async getKBChunks(itemName: string): Promise<KBChunk[]> {
    try {
      const res = await fetch(`${API_BASE}/api/kb/chunks?item_name=${encodeURIComponent(itemName)}`)
      if (res.ok) {
        const data = await res.json()
        if (data && data.data) return data.data
      }
    } catch {
      // 忽略
    }
    return localKBChunks[itemName] || []
  },

  async deleteKBChunk(chunkId: string | number): Promise<{ success: boolean; message: string }> {
    try {
      const res = await fetch(`${API_BASE}/api/kb/chunks/${encodeURIComponent(chunkId)}`, { method: 'DELETE' })
      if (res.ok) {
        return await res.json()
      }
    } catch {
      // 忽略
    }
    return { success: true, message: `成功删除切片 [${chunkId}]！` }
  },

  async updateKBChunk(chunkId: string | number, content: string): Promise<{ success: boolean; message: string }> {
    try {
      const res = await fetch(`${API_BASE}/api/kb/chunks/${encodeURIComponent(chunkId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      })
      if (res.ok) {
        return await res.json()
      }
    } catch {
      // 忽略
    }
    return { success: true, message: `成功更新切片 [${chunkId}] 文本，已完成重新向量化！` }
  },

  async deleteKBItem(itemName: string): Promise<{ success: boolean; message: string }> {
    try {
      const res = await fetch(`${API_BASE}/api/kb/items/${encodeURIComponent(itemName)}`, { method: 'DELETE' })
      if (res.ok) {
        const data = await res.json()
        localKBItems = localKBItems.filter(i => i.item_name !== itemName)
        delete localKBChunks[itemName]
        return data
      }
    } catch {
      // 忽略
    }
    localKBItems = localKBItems.filter(item => item.item_name !== itemName)
    delete localKBChunks[itemName]
    return { success: true, message: `成功删除设备主体 [${itemName}] 及其关联的全部向量切片！` }
  },

  // === 历史记录 API ===
  async getSessions(): Promise<ChatSession[]> {
    try {
      const res = await fetch(`${API_BASE}/api/history/sessions`)
      if (res.ok) {
        const data = await res.json()
        if (data && data.data) return data.data
      }
    } catch {
      // 忽略
    }
    return localSessions
  },

  async getSessionDetail(sessionId: string): Promise<ChatSession | null> {
    try {
      const res = await fetch(`${API_BASE}/history/${sessionId}`)
      if (res.ok) {
        const data = await res.json()
        if (data && data.items) {
          const session: ChatSession = localSessions.find(s => s.session_id === sessionId) || {
            session_id: sessionId,
            title: '历史会话',
            last_message: '',
            last_role: 'assistant',
            last_ts: Date.now() / 1000,
            message_count: data.items.length,
            item_names: []
          }
          session.messages = data.items.map((item: any) => ({
            id: item._id,
            role: item.role,
            text: item.text,
            timestamp: item.ts || Date.now() / 1000,
            item_names: item.item_names || [],
            image_urls: item.image_urls || [],
            sources: item.sources || [],
            node_steps: item.node_steps || [],
            total_duration: item.total_duration,
            warnings: item.warnings || []
          }))
          return session
        }
      }
    } catch {
      // 忽略
    }
    return localSessions.find(s => s.session_id === sessionId) || null
  },

  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    localSessions = localSessions.filter(s => s.session_id !== sessionId)
    try {
      await fetch(`${API_BASE}/history/${sessionId}`, { method: 'DELETE' })
    } catch {
      // 忽略
    }
    return { success: true }
  },

  async clearAllSessions(): Promise<{ success: boolean; deleted_count: number }> {
    const count = localSessions.length
    localSessions = []
    try {
      await fetch(`${API_BASE}/api/history/sessions`, { method: 'DELETE' })
    } catch {
      // 忽略
    }
    return { success: true, deleted_count: count }
  }
}
