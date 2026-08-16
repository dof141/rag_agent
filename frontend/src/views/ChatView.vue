<template>
  <div class="chat-container">
    <!-- Left Session Selector Sidebar -->
    <div class="sessions-sidebar">
      <div class="sidebar-header">
        <button class="new-chat-btn" @click="startNewChat">
          <Plus class="icon" />
          <span>新建问答会话</span>
        </button>
      </div>

      <div class="sessions-list">
        <div 
          v-for="s in sessions" 
          :key="s.session_id" 
          class="session-item"
          :class="{ 'active': currentSessionId === s.session_id }"
          @click="selectSession(s.session_id)"
        >
          <MessageSquare class="session-icon" />
          <div class="session-info">
            <span class="session-title">{{ s.title }}</span>
            <span class="session-snippet">{{ s.last_message }}</span>
          </div>
          <button class="del-btn" @click.stop="deleteSession(s.session_id)" title="删除会话">
            <Trash2 class="trash-icon" />
          </button>
        </div>
      </div>
    </div>

    <!-- Main Chat Workspace -->
    <div class="chat-workspace">
      <!-- Top Sample Question Chips -->
      <div class="sample-chips">
        <span class="chip-label">💡 常见示例提问：</span>
        <button 
          v-for="q in sampleQuestions" 
          :key="q" 
          class="sample-chip"
          @click="useSampleQuestion(q)"
        >
          {{ q }}
        </button>

        <div class="stream-mode-container">
          <button 
            class="stream-toggle-btn"
            :class="{ active: isStreamMode }"
            @click="toggleStreamMode"
            :title="isStreamMode ? '当前：打字机流式输出模式 (可实时展示回答及节点耗时)' : '当前：阻塞全量模式 (等待全部节点执行完成一次性返回)'"
          >
            <Zap class="mode-icon" v-if="isStreamMode" />
            <Box class="mode-icon" v-else />
            <span>{{ isStreamMode ? '⚡ 打字机流式模式' : '📦 阻塞全量模式' }}</span>
          </button>
        </div>
      </div>

      <!-- Messages Stream Viewport -->
      <div class="messages-viewport" ref="viewportRef">
        <div v-if="messages.length === 0" class="empty-placeholder">
          <div class="welcome-box">
            <Bot class="bot-hero-icon" />
            <h3>欢迎使用 RAG Agent 智能设备问答助手</h3>
            <p>基于 LangGraph 状态机与 BGE-M3 混合检索，输入您的设备型号或故障描述即可获取精准回答。</p>
          </div>
        </div>

        <div 
          v-for="msg in messages" 
          :key="msg.id" 
          class="message-row"
          :class="msg.role"
        >
          <div class="msg-avatar">
            <User v-if="msg.role === 'user'" />
            <Bot v-else />
          </div>

          <div class="msg-card">
            <div class="msg-meta">
              <span class="sender-name">{{ msg.role === 'user' ? '用户提问' : 'RAG Agent' }}</span>
              <span class="time">{{ new Date(msg.timestamp * 1000).toLocaleTimeString() }}</span>
            </div>

            <!-- Collapsible Node Execution Process Panel (Positioned ABOVE Message Body) -->
            <div v-if="msg.role === 'assistant' && msg.node_steps && msg.node_steps.length" class="collapsible-nodes-panel">
              <div 
                class="nodes-panel-header" 
                @click="msg.isStepsCollapsed = !msg.isStepsCollapsed"
                title="点击展开/折叠节点思考与检索全流程"
              >
                <div class="header-title">
                  <ChevronDown v-if="!msg.isStepsCollapsed" class="toggle-icon" />
                  <ChevronRight v-else class="toggle-icon" />
                  <Activity class="timeline-icon" />
                  <span>思考与知识库检索节点 ({{ getCompletedCount(msg.node_steps) }}/{{ msg.node_steps.length }})</span>
                  <span v-if="msg.total_duration" class="total-duration-tag">⚡ 耗时 {{ msg.total_duration }}s</span>
                </div>
                <span class="status-tag">已完成</span>
              </div>

              <!-- Collapsible Stepper Chips -->
              <div v-show="!msg.isStepsCollapsed" class="nodes-stepper-body">
                <div 
                  v-for="step in msg.node_steps" 
                  :key="step.node_id"
                  class="node-chip"
                  :class="step.status"
                >
                  <Check v-if="step.status === 'completed'" class="step-status-icon" />
                  <Loader2 v-else-if="step.status === 'running'" class="step-status-icon spin" />
                  <span v-else class="node-num">•</span>
                  <span>{{ step.name }}</span>
                  <span v-if="step.duration !== undefined" class="node-duration-tag">{{ step.duration }}s</span>
                </div>
              </div>
            </div>

            <!-- Markdown Answer Content + Smart Auto Image Rendering -->
            <div class="msg-body-wrapper">
              <div class="msg-body markdown-body" v-html="renderMarkdown(msg.text)"></div>
              <span v-if="msg.isStreaming" class="typewriter-cursor">▌</span>
            </div>

            <div
              v-if="msg.role === 'assistant' && msg.warnings?.length"
              class="message-warnings"
              role="status"
              aria-live="polite"
            >
              <div v-for="warning in msg.warnings" :key="warning.code" class="message-warning">
                <TriangleAlert class="warning-icon" />
                <span>{{ warning.message }}</span>
              </div>
            </div>

            <!-- Citations & Sources (Recall Sources) -->
            <div v-if="msg.sources && msg.sources.length" class="sources-panel">
              <div class="sources-header">
                <FileText class="s-icon" />
                <span>知识库引用来源 (Recall Sources):</span>
              </div>
              <div class="sources-grid">
                <div 
                  v-for="s in msg.sources" 
                  :key="s.chunk_id" 
                  class="source-card"
                  @click="openChunkDrawer(s)"
                  title="点击查看切片文本与 1024 维向量详情"
                >
                  <div class="source-card-main">
                    <span class="s-file-title">【相关文档来源】：{{ s.file_title || s.title }} <template v-if="s.parent_title">({{ s.parent_title }})</template></span>
                    <span class="s-item-tag" v-if="s.item_name">🏷️ {{ s.item_name }}</span>
                  </div>
                  <div class="source-card-badge">
                    <span class="s-score">相似度 {{ (s.score && s.score > 1 ? s.score : (s.score ?? 0.942) * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Thinking Loading Box with Real Node Status Tracker (仅在阻塞全量模式下显示) -->
        <div v-if="isThinking && !isStreamMode" class="message-row assistant">
          <div class="msg-avatar"><Bot /></div>
          <div class="msg-card thinking-card-box">
            <div class="thinking-header">
              <div class="typing-indicator">
                <span></span><span></span><span></span>
              </div>
              <span class="thinking-text">{{ activeNodeText }}</span>
            </div>

            <!-- Execution Node Chips Stream -->
            <div class="nodes-stepper live">
              <div 
                v-for="step in activeNodeSteps" 
                :key="step.node_id"
                class="node-chip"
                :class="step.status"
              >
                <Check v-if="step.status === 'completed'" class="step-status-icon" />
                <Loader2 v-else-if="step.status === 'running'" class="step-status-icon spin" />
                <span v-else class="node-num">•</span>
                <span>{{ step.name }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Area with Inline Candidate Selector Toolbar right above Input Box -->
      <div class="input-area">
        <!-- Inline Candidates Panel (Pops up directly above input box) -->
        <div v-if="showCandidateSelector" class="inline-candidate-panel">
          <div class="panel-header">
            <Cpu class="panel-icon" />
            <span>❓ 检测到多个相关笔记分类/学习主题，请选择您想查询的具体主题：</span>
          </div>

          <div class="candidates-inline-grid">
            <button 
              v-for="item in activeCandidates" 
              :key="item.id" 
              class="candidate-chip-btn"
              @click="onCandidateConfirmed(item.id)"
            >
              <span class="chip-item-name">{{ item.item_name }}</span>
              <span class="chip-file-title" v-if="item.file_title">({{ item.file_title }})</span>
              <span class="chip-score" v-if="item.score">匹配度 {{ (item.score * 100).toFixed(0) }}%</span>
            </button>
          </div>
        </div>

        <div class="input-box">
          <textarea 
            v-model="inputQuery"
            placeholder="输入笔记主题、技术名词或学习疑问 (Enter 发送, Shift+Enter 换行)..."
            @keydown.enter.prevent="handleEnter"
            rows="2"
          ></textarea>

          <button 
            class="send-btn" 
            :disabled="!inputQuery.trim() || isThinking"
            @click="sendUserMessage"
          >
            <Send class="send-icon" />
            <span>发送</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Chunk Detail Drawer Modal -->
    <ChunkDetailDrawer v-model="drawerVisible" :chunk="selectedChunk" />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { Plus, MessageSquare, Trash2, Bot, User, Send, FileText, Activity, Check, Loader2, Cpu, ChevronDown, ChevronRight, Zap, Box, TriangleAlert } from 'lucide-vue-next'
import type { ChatSession, ChatMessage, ChatWarning, CandidateItem, ChunkSource, KBChunk } from '../types'
import { api } from '../services/api'
import { streamSse, type SseEvent } from '../services/sse'
import { renderMarkdown } from '../utils/markdown'
import ChunkDetailDrawer from '../components/ChunkDetailDrawer.vue'

const getCompletedCount = (steps?: NodeStep[]) => {
  if (!steps) return 0
  return steps.filter(s => s.status === 'completed').length
}

interface NodeStep {
  node_id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'paused'
  duration?: number
}

const REAL_BACKEND_NODES: NodeStep[] = [
  { node_id: 'node_item_name_confirm', name: '确认学习主题', status: 'pending' },
  { node_id: 'node_search_embedding', name: '切片搜索', status: 'pending' },
  { node_id: 'node_search_embedding_hyde', name: '切片搜索(假设性文档)', status: 'pending' },
  { node_id: 'node_web_search_mcp', name: '网络搜索', status: 'pending' },
  { node_id: 'node_rrf', name: '倒排融合', status: 'pending' },
  { node_id: 'node_rerank', name: '重排序', status: 'pending' },
  { node_id: 'node_answer_output', name: '生成答案', status: 'pending' }
]

const mergeWarnings = (
  current: ChatWarning[] = [],
  incoming: ChatWarning[] = []
) => {
  const byCode = new Map(current.map(warning => [warning.code, warning]))
  incoming.forEach(warning => byCode.set(warning.code, warning))
  return Array.from(byCode.values())
}

const warningsFromPayload = (payload: any): ChatWarning[] => {
  if (Array.isArray(payload?.warnings)) {
    return payload.warnings.filter(
      (warning: any) => typeof warning?.code === 'string' && typeof warning?.message === 'string'
    )
  }
  if (typeof payload?.code === 'string' && typeof payload?.message === 'string') {
    return [{ code: payload.code, message: payload.message }]
  }
  return []
}

const mergeMessageWarnings = (message: ChatMessage | undefined, payload: any) => {
  if (!message) return
  message.warnings = mergeWarnings(message.warnings, warningsFromPayload(payload))
}

const sessions = ref<ChatSession[]>([])
const currentSessionId = ref<string>('')
const messages = ref<ChatMessage[]>([])
const inputQuery = ref<string>('')
const isThinking = ref<boolean>(false)
const activeNodeText = ref<string>('正在启动 LangGraph 节点处理流...')
const activeNodeSteps = ref<NodeStep[]>([])

const isStreamMode = ref<boolean>(localStorage.getItem('rag_is_stream') !== 'false')

const toggleStreamMode = () => {
  isStreamMode.value = !isStreamMode.value
  localStorage.setItem('rag_is_stream', String(isStreamMode.value))
}

const showCandidateSelector = ref<boolean>(false)
const activeCandidates = ref<CandidateItem[]>([])
const pendingRequestQuery = ref<string>('')
const pendingRequestId = ref<string>('')

const viewportRef = ref<HTMLElement | null>(null)

const drawerVisible = ref(false)
const selectedChunk = ref<KBChunk | null>(null)

const openChunkDrawer = (s: ChunkSource) => {
  selectedChunk.value = {
    chunk_id: s.chunk_id,
    file_title: s.file_title || s.title || '学习资料手册.pdf',
    item_name: s.item_name || '个人学习笔记',
    title: s.title || s.file_title || '章节说明',
    parent_title: s.parent_title || '# 章节内容',
    part: 1,
    content: s.content || '暂无详细切片文本内容。'
  }
  drawerVisible.value = true
}

const sampleQuestions = [
  'Neo4j 数据库安装与使用',
  '大模型 RAG 架构与向量检索原理',
  'Python 高效并发与异步 IO 实战'
]

onMounted(async () => {
  await loadSessions()
})

const loadSessions = async () => {
  sessions.value = await api.getSessions()
  if (sessions.value.length > 0 && !currentSessionId.value) {
    selectSession(sessions.value[0].session_id)
  }
}

const selectSession = async (sid: string) => {
  currentSessionId.value = sid
  const session = await api.getSessionDetail(sid)
  if (session && session.messages) {
    messages.value = session.messages
  } else {
    messages.value = []
  }
  scrollToBottom()
}

const startNewChat = () => {
  const newSid = `sess-${Date.now()}`
  currentSessionId.value = newSid
  messages.value = []
}

const deleteSession = async (sid: string) => {
  await api.deleteSession(sid)
  await loadSessions()
  if (currentSessionId.value === sid) {
    startNewChat()
  }
}

const useSampleQuestion = (q: string) => {
  inputQuery.value = q
  sendUserMessage()
}

const handleEnter = (e: KeyboardEvent) => {
  if (!e.shiftKey) {
    sendUserMessage()
  }
}

const normalizeCandidates = (candidates: any[] = []): CandidateItem[] => candidates.map(candidate => {
  if (typeof candidate === 'string') {
    return { id: candidate, item_name: candidate }
  }
  const id = String(candidate.id ?? candidate.item_name)
  return {
    id,
    item_name: candidate.item_name || id,
    file_title: candidate.file_title,
    score: candidate.score,
    description: candidate.description,
  }
})

const showConfirmation = (payload: any, query: string) => {
  pendingRequestId.value = payload.request_id || pendingRequestId.value
  activeCandidates.value = normalizeCandidates(payload.candidates || [])
  pendingRequestQuery.value = query
  showCandidateSelector.value = true
  activeNodeSteps.value[0].status = 'paused'
  isThinking.value = false
}

const appendAssistantMessage = (text: string, extras: Partial<ChatMessage> = {}) => {
  messages.value.push({
    id: `m-ast-${Date.now()}`,
    role: 'assistant',
    text,
    timestamp: Date.now() / 1000,
    node_steps: JSON.parse(JSON.stringify(activeNodeSteps.value)),
    ...extras,
  })
  scrollToBottom()
}

const errorMessage = (error: unknown) => error instanceof Error ? error.message : '问答请求失败'

const applyProgress = (payload: any, message: ChatMessage) => {
  const doneList: string[] = payload.done_list || []
  const runningList: string[] = payload.running_list || []
  const nodeDurations = payload.node_durations || {}
  activeNodeSteps.value.forEach(node => {
    if (doneList.some(value => value === node.name || value === node.node_id)) node.status = 'completed'
    else if (runningList.some(value => value === node.name || value === node.node_id)) node.status = 'running'
    if (nodeDurations[node.node_id] !== undefined) node.duration = nodeDurations[node.node_id]
  })
  if (runningList.length) activeNodeText.value = `后端正在运行: ${runningList.join(', ')}`
  message.node_steps = JSON.parse(JSON.stringify(activeNodeSteps.value))
}

const consumeAnswerStream = async (requestId: string, query: string) => {
  const assistantMsgIndex = messages.value.length
  appendAssistantMessage('', { isStreaming: true })
  let terminalReceived = false

  try {
    await streamSse(`/stream/${encodeURIComponent(requestId)}`, {}, async (event: SseEvent) => {
      const message = messages.value[assistantMsgIndex]
      if (!message) return

      if (event.type === 'delta') {
        if (typeof event.data.delta === 'string') message.text += event.data.delta
      } else if (event.type === 'progress') {
        applyProgress(event.data, message)
      } else if (event.type === 'warning') {
        mergeMessageWarnings(message, event.data)
      } else if (event.type === 'confirmation_required') {
        terminalReceived = true
        showConfirmation(event.data, query)
        if (!message.text) messages.value.splice(assistantMsgIndex, 1)
      } else if (event.type === 'final') {
        terminalReceived = true
        if (typeof event.data.answer === 'string') message.text = event.data.answer
        if (Array.isArray(event.data.sources)) message.sources = event.data.sources
        if (Array.isArray(event.data.image_urls)) message.image_urls = event.data.image_urls
        if (typeof event.data.total_duration === 'number') message.total_duration = event.data.total_duration
        if (Array.isArray(event.data.node_steps)) message.node_steps = event.data.node_steps
        mergeMessageWarnings(message, event.data)
        activeNodeSteps.value.forEach(node => node.status = 'completed')
        message.isStreaming = false
        isThinking.value = false
        await loadSessions().catch(() => undefined)
      } else if (event.type === 'error') {
        terminalReceived = true
        message.text = typeof event.data.message === 'string' ? event.data.message : '问答请求失败'
        message.isStreaming = false
        isThinking.value = false
      }
      scrollToBottom()
    })

    if (!terminalReceived) throw new Error('问答流意外中断')
  } catch (error) {
    const message = messages.value[assistantMsgIndex]
    if (message) {
      message.text = errorMessage(error)
      message.isStreaming = false
    } else {
      appendAssistantMessage(errorMessage(error))
    }
    isThinking.value = false
    scrollToBottom()
  }
}

const sendUserMessage = async () => {
  const query = inputQuery.value.trim()
  if (!query || isThinking.value) return

  inputQuery.value = ''
  isThinking.value = true
  showCandidateSelector.value = false
  activeNodeText.value = '正在连通 LangGraph 图服务...'

  const sid = currentSessionId.value || `sess-${Date.now()}`
  currentSessionId.value = sid

  messages.value.push({
    id: `m-usr-${Date.now()}`,
    role: 'user',
    text: query,
    timestamp: Date.now() / 1000
  })
  scrollToBottom()

  activeNodeSteps.value = REAL_BACKEND_NODES.map(n => ({ ...n, status: 'pending', duration: undefined }))
  activeNodeSteps.value[0].status = 'running'

  try {
    const isStream = isStreamMode.value
    const response = await api.sendQuery(query, sid, isStream)
    pendingRequestId.value = response.request_id

    if (response.status === 'confirmation_required') {
      showConfirmation(response, query)
      return
    }
    if (isStream) {
      await consumeAnswerStream(response.request_id, query)
      return
    }
    if (response.status !== 'final' || !response.answer) throw new Error('问答流程未产生有效答案')
    activeNodeSteps.value.forEach(node => node.status = 'completed')
    appendAssistantMessage(response.answer)
    isThinking.value = false
    await loadSessions().catch(() => undefined)
  } catch (error) {
    appendAssistantMessage(errorMessage(error))
    isThinking.value = false
  }
}

const onCandidateConfirmed = async (candidateId: string) => {
  const candidate = activeCandidates.value.find(item => item.id === candidateId)
  const candidateLabel = candidate?.item_name || candidateId
  showCandidateSelector.value = false
  isThinking.value = true
  activeNodeText.value = `已确认学习主题 [${candidateLabel}]，正在恢复后端 LangGraph 状态机...`

  activeNodeSteps.value[0].status = 'completed'
  activeNodeSteps.value[1].status = 'running'

  try {
    const response = await api.confirmQuery(
      currentSessionId.value,
      pendingRequestId.value,
      candidateId,
    )
    pendingRequestId.value = response.request_id
    if (response.status === 'confirmation_required') {
      showConfirmation(response, pendingRequestQuery.value)
      return
    }
    if (response.status === 'processing') {
      await consumeAnswerStream(response.request_id, pendingRequestQuery.value)
      return
    }
    if (response.status !== 'final' || !response.answer) throw new Error('问答流程未产生有效答案')
    activeNodeSteps.value.forEach(node => node.status = 'completed')
    appendAssistantMessage(response.answer)
    isThinking.value = false
    await loadSessions().catch(() => undefined)
  } catch (error) {
    appendAssistantMessage(errorMessage(error))
    isThinking.value = false
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (viewportRef.value) {
      viewportRef.value.scrollTop = viewportRef.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.chat-container {
  display: flex;
  height: calc(100vh - 64px);
  overflow: hidden;
}

.sessions-sidebar {
  width: 260px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 14px;
  border-bottom: 1px solid var(--border-color);
}

.new-chat-btn {
  width: 100%;
  padding: 10px;
  border-radius: 10px;
  background: var(--accent-gradient);
  color: white;
  border: none;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
  transition: transform 0.2s ease;
}

.new-chat-btn:hover {
  transform: translateY(-1px);
}

.sessions-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.session-item {
  display: flex;
  align-items: center;
  padding: 10px 12px;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  gap: 10px;
  position: relative;
}

.session-item:hover {
  background: var(--bg-hover);
}

.session-item.active {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
}

.session-icon {
  width: 18px;
  height: 18px;
  color: #a855f7;
  flex-shrink: 0;
}

.session-info {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
}

.session-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-snippet {
  font-size: 0.72rem;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.del-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.session-item:hover .del-btn {
  opacity: 1;
}

.del-btn:hover {
  color: #ef4444;
}

.trash-icon {
  width: 14px;
  height: 14px;
}

/* Chat Workspace */
.chat-workspace {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg-main);
}

.sample-chips {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  gap: 10px;
  overflow-x: auto;
  background: var(--bg-header);
}

.chip-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  flex-shrink: 0;
}

.sample-chip {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 0.78rem;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s ease;
}

.sample-chip:hover {
  background: rgba(168, 85, 247, 0.15);
  border-color: #a855f7;
  color: #c084fc;
}

.stream-mode-container {
  margin-left: auto;
  flex-shrink: 0;
}

.stream-toggle-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 0.78rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.25s ease;
}

.stream-toggle-btn.active {
  background: rgba(168, 85, 247, 0.15);
  border-color: rgba(168, 85, 247, 0.4);
  color: #c084fc;
  box-shadow: 0 0 10px rgba(168, 85, 247, 0.2);
}

.stream-toggle-btn:hover {
  border-color: #a855f7;
}

.mode-icon {
  width: 14px;
  height: 14px;
}

.msg-body-wrapper {
  position: relative;
  display: inline-block;
  width: 100%;
}

.typewriter-cursor {
  display: inline-block;
  color: #a855f7;
  font-weight: bold;
  margin-left: 2px;
  font-size: 1.1rem;
  animation: blink 0.8s infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.messages-viewport {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.empty-placeholder {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  text-align: center;
}

.welcome-box {
  max-width: 480px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
}

.bot-hero-icon {
  width: 56px;
  height: 56px;
  color: #6366f1;
}

.message-row {
  display: flex;
  gap: 14px;
  max-width: 85%;
}

.message-row.user {
  margin-left: auto;
  flex-direction: row-reverse;
}

.msg-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-primary);
  flex-shrink: 0;
}

.message-row.user .msg-avatar {
  background: linear-gradient(135deg, #6366f1, #a855f7);
  color: white;
}

.msg-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 14px 18px;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
}

.message-row.user .msg-card {
  background: var(--accent-gradient);
  color: white;
  border: none;
}

.msg-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.message-row.user .msg-meta {
  color: rgba(255, 255, 255, 0.8);
}

/* Node Timeline Styling */
.node-timeline-box {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 8px 12px;
  margin-bottom: 10px;
}

.timeline-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: #a855f7;
  font-weight: 600;
  margin-bottom: 6px;
}

.timeline-icon { width: 14px; height: 14px; }

.nodes-stepper {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.nodes-stepper.live {
  margin-top: 8px;
}

.node-chip {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 0.72rem;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
}

.node-chip.completed {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
  border-color: rgba(34, 197, 94, 0.3);
}

.node-chip.running {
  background: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
  border-color: rgba(59, 130, 246, 0.4);
}

.node-chip.paused {
  background: rgba(249, 115, 22, 0.15);
  color: #fb923c;
  border-color: rgba(249, 115, 22, 0.4);
}

.step-status-icon { width: 12px; height: 12px; }
.step-status-icon.spin { animation: spin 1s linear infinite; }

.total-duration-tag {
  font-size: 0.7rem;
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  padding: 1px 6px;
  border-radius: 999px;
  border: 1px solid rgba(168, 85, 247, 0.3);
  margin-left: 6px;
}

.node-duration-tag {
  font-size: 0.68rem;
  opacity: 0.85;
  font-family: monospace;
  background: rgba(255, 255, 255, 0.1);
  padding: 0 4px;
  border-radius: 4px;
  margin-left: 2px;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.markdown-body {
  font-size: 0.92rem;
  line-height: 1.65;
}

/* 关键强制逻辑：美化内联直接展示的渲染图片 */
.markdown-body :deep(img) {
  max-width: 100%;
  border-radius: 12px;
  margin: 12px 0;
  border: 1px solid var(--border-color);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
  display: block;
}

.message-warnings {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 12px;
  min-width: 0;
}

.message-warning {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  min-width: 0;
  padding: 8px 10px;
  border: 1px solid rgba(202, 138, 4, 0.48);
  border-radius: 6px;
  background: rgba(250, 204, 21, 0.12);
  color: #a16207;
  font-size: 0.78rem;
  line-height: 1.5;
}

.message-warning span {
  min-width: 0;
  overflow-wrap: anywhere;
}

.warning-icon {
  width: 16px;
  height: 16px;
  margin-top: 1px;
  flex: 0 0 16px;
  color: #ca8a04;
}

:global(html.dark .message-warning) {
  background: rgba(250, 204, 21, 0.09);
  color: #fde68a;
}

@media (max-width: 768px) {
  :global(.app-layout > .sidebar-container),
  .sessions-sidebar,
  :global(.header-right .model-badge) {
    display: none;
  }

  :global(.main-content),
  .chat-workspace {
    width: 100%;
    min-width: 0;
  }

  :global(.header-bar) {
    height: 56px;
    padding: 0 14px;
  }

  :global(.header-bar .page-title) {
    font-size: 1rem;
  }

  .chat-container {
    width: 100%;
    height: calc(100vh - 56px);
  }

  .sample-chips {
    min-height: 44px;
    padding: 8px 12px;
    justify-content: flex-end;
  }

  .sample-chips > .chip-label,
  .sample-chips > .sample-chip {
    display: none;
  }

  .messages-viewport {
    padding: 12px;
    gap: 16px;
  }

  .message-row {
    width: 100%;
    max-width: 100%;
    gap: 8px;
  }

  .message-row.user {
    width: auto;
    max-width: 92%;
  }

  .msg-avatar {
    width: 32px;
    height: 32px;
  }

  .message-row.assistant .msg-card {
    flex: 1;
  }

  .msg-card {
    min-width: 0;
    max-width: calc(100% - 40px);
    padding: 12px;
    border-radius: 8px;
  }

  .input-area {
    padding: 10px 12px;
  }

  .input-box {
    min-width: 0;
    padding: 8px 10px;
  }

  .input-box textarea {
    min-width: 0;
  }

  .send-btn {
    padding: 8px;
  }

  .send-btn span {
    display: none;
  }
}

.sources-panel {
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid var(--border-color);
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  color: #a855f7;
  font-weight: 600;
  margin-bottom: 8px;
}

.s-icon { width: 14px; height: 14px; }

.sources-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.source-card {
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
}

.s-title { font-weight: 600; color: var(--text-primary); }
.s-score { color: #22c55e; font-weight: 600; }

.thinking-card-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.thinking-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.thinking-text { font-size: 0.82rem; color: var(--text-secondary); }

.typing-indicator span {
  display: inline-block;
  width: 6px;
  height: 6px;
  background: #6366f1;
  border-radius: 50%;
  margin-right: 3px;
  animation: bounce 1.4s infinite ease-in-out both;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

/* Input Area with Inline Candidate Selector Panel */
.input-area {
  padding: 16px 24px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-header);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Inline Candidate Panel (Above Input Box) */
.inline-candidate-panel {
  background: rgba(99, 102, 241, 0.12);
  border: 1px solid rgba(99, 102, 241, 0.35);
  border-radius: 14px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  animation: slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-primary);
}

.panel-icon {
  width: 18px;
  height: 18px;
  color: #6366f1;
  flex-shrink: 0;
}

.candidates-inline-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.candidate-chip-btn {
  background: var(--bg-card);
  border: 1px solid rgba(99, 102, 241, 0.4);
  border-radius: 10px;
  padding: 8px 14px;
  color: var(--text-primary);
  font-size: 0.82rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.candidate-chip-btn:hover {
  background: var(--accent-gradient);
  color: white;
  border-color: transparent;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.chip-item-name {
  font-weight: 700;
}

.chip-file-title {
  font-size: 0.75rem;
  opacity: 0.8;
}

.chip-score {
  background: rgba(34, 197, 94, 0.2);
  color: #4ade80;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
}

.candidate-chip-btn:hover .chip-score {
  background: rgba(255, 255, 255, 0.25);
  color: white;
}

.input-box {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 10px 14px;
}

.input-box textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  resize: none;
  color: var(--text-primary);
  font-family: inherit;
  font-size: 0.92rem;
}

.send-btn {
  background: var(--accent-gradient);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 8px 16px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: opacity 0.2s ease;
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.send-icon { width: 16px; height: 16px; }

/* Collapsible Nodes Panel (Positioned Above Message Body) */
.collapsible-nodes-panel {
  background: rgba(168, 85, 247, 0.06);
  border: 1px solid rgba(168, 85, 247, 0.22);
  border-radius: 10px;
  margin-bottom: 12px;
  overflow: hidden;
  transition: all 0.2s ease;
}

.nodes-panel-header {
  padding: 8px 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
  background: rgba(168, 85, 247, 0.04);
  transition: background 0.2s ease;
}

.nodes-panel-header:hover {
  background: rgba(168, 85, 247, 0.12);
}

.header-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  font-weight: 600;
  color: #a855f7;
}

.toggle-icon {
  width: 14px;
  height: 14px;
  color: #a855f7;
  transition: transform 0.2s ease;
}

.status-tag {
  font-size: 0.72rem;
  font-weight: 600;
  color: #16a34a;
  background: rgba(34, 197, 94, 0.15);
  padding: 2px 8px;
  border-radius: 6px;
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.nodes-stepper-body {
  padding: 10px 12px;
  border-top: 1px dashed rgba(168, 85, 247, 0.2);
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

/* Sources Panel & Recall Cards Styling */
.sources-panel {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sources-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.82rem;
  font-weight: 700;
  color: #a855f7;
}

.s-icon {
  width: 16px;
  height: 16px;
  color: #a855f7;
}

.sources-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.source-card {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 10px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.source-card:hover {
  background: var(--bg-card);
  border-color: #a855f7;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(168, 85, 247, 0.15);
}

.source-card-main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;
}

.s-file-title {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--text-primary);
  word-break: break-all;
}

.s-item-tag {
  font-size: 0.76rem;
  color: #6366f1;
  font-weight: 500;
}

.source-card-badge {
  flex-shrink: 0;
}

.s-score {
  background: rgba(34, 197, 94, 0.15);
  color: #16a34a;
  border: 1px solid rgba(34, 197, 94, 0.3);
  font-size: 0.78rem;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 6px;
}
</style>
