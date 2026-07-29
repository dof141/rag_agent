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
                </div>
              </div>
            </div>

            <!-- Markdown Answer Content + Smart Auto Image Rendering -->
            <div class="msg-body markdown-body" v-html="renderMarkdown(msg.text)"></div>

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

        <!-- Thinking Loading Box with Real Node Status Tracker -->
        <div v-if="isThinking" class="message-row assistant">
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
              @click="onCandidateConfirmed(item.item_name)"
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
import { Plus, MessageSquare, Trash2, Bot, User, Send, FileText, Activity, Check, Loader2, Cpu, ChevronDown, ChevronRight } from 'lucide-vue-next'
import type { ChatSession, ChatMessage, CandidateItem, ChunkSource, KBChunk } from '../types'
import { api } from '../services/api'
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

const sessions = ref<ChatSession[]>([])
const currentSessionId = ref<string>('')
const messages = ref<ChatMessage[]>([])
const inputQuery = ref<string>('')
const isThinking = ref<boolean>(false)
const activeNodeText = ref<string>('正在启动 LangGraph 节点处理流...')
const activeNodeSteps = ref<NodeStep[]>([])

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

const pollTaskNodeStatus = async (taskId: string): Promise<boolean> => {
  const statusRes = await api.getTaskStatus(taskId)
  const doneList = statusRes.done_list || []
  const runningList = statusRes.running_list || []
  const currentGlobalStatus = statusRes.status

  activeNodeSteps.value.forEach(node => {
    const cnName = node.name
    // 全精确比对，防止子串“切片搜索”误将“切片搜索(假设性文档)”点亮
    const isDone = doneList.some((d: string) => d === cnName || d === node.node_id)
    const isRunning = runningList.some((r: string) => r === cnName || r === node.node_id)

    if (isDone) {
      node.status = 'completed'
    } else if (isRunning) {
      node.status = 'running'
    }
  })

  if (runningList.length > 0) {
    activeNodeText.value = `后端正在运行: ${runningList.join(', ')}`
  }

  if (currentGlobalStatus === 'waiting_confirmation') {
    activeNodeSteps.value[0].status = 'paused'
    return true
  }

  if (currentGlobalStatus === 'completed' || doneList.includes('生成答案') || doneList.includes('node_answer_output')) {
    activeNodeSteps.value.forEach(n => n.status = 'completed')
    return false
  }

  return false
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

  activeNodeSteps.value = REAL_BACKEND_NODES.map(n => ({ ...n, status: 'pending' }))
  activeNodeSteps.value[0].status = 'running'

  const res = await api.sendQuery(query, sid)
  pendingRequestId.value = res.request_id || ''

  if (res.awaiting_confirmation && res.candidates && res.candidates.length > 0) {
    isThinking.value = false
    activeNodeSteps.value[0].status = 'paused'
    
    activeCandidates.value = res.candidates.map((c: any) => {
      if (typeof c === 'string') return { id: c, item_name: c, file_title: `${c} 设备手册` }
      return { id: c.id || c.item_name, item_name: c.item_name || c.id, file_title: c.file_title || `${c.item_name} 手册`, score: c.score }
    })

    pendingRequestQuery.value = query
    showCandidateSelector.value = true
    return
  }

  let isDone = false
  let pollAttempts = 0
  while (!isDone && pollAttempts < 300) {
    await new Promise(r => setTimeout(r, 500))
    pollAttempts++
    const isPaused = await pollTaskNodeStatus(res.request_id)
    if (isPaused) {
      isThinking.value = false
      showCandidateSelector.value = true
      return
    }
    if (activeNodeSteps.value.every(n => n.status === 'completed')) {
      isDone = true
    }
  }

  isThinking.value = false

  // 从后端同步最新的完整会话记录（重试等待 MongoDB 异步落盘完成）
  let sessionDetail = await api.getSessionDetail(sid)
  let lastMsg = sessionDetail?.messages?.length ? sessionDetail.messages[sessionDetail.messages.length - 1] : null

  let retries = 0
  while ((!lastMsg || lastMsg.role !== 'assistant') && retries < 20) {
    await new Promise(r => setTimeout(r, 400))
    retries++
    sessionDetail = await api.getSessionDetail(sid)
    lastMsg = sessionDetail?.messages?.length ? sessionDetail.messages[sessionDetail.messages.length - 1] : null
  }

  if (sessionDetail && sessionDetail.messages && sessionDetail.messages.length > 0) {
    const lastMessage = sessionDetail.messages[sessionDetail.messages.length - 1]
    if (lastMessage.role === 'assistant') {
      messages.value = sessionDetail.messages
      messages.value[messages.value.length - 1].node_steps = JSON.parse(JSON.stringify(activeNodeSteps.value))
    }
  }
  scrollToBottom()
  await loadSessions()
}

const onCandidateConfirmed = async (candidateName: string) => {
  showCandidateSelector.value = false
  isThinking.value = true
  activeNodeText.value = `已确认学习主题 [${candidateName}]，正在恢复后端 LangGraph 状态机...`

  activeNodeSteps.value[0].status = 'completed'
  activeNodeSteps.value[1].status = 'running'

  let newReqId = pendingRequestId.value
  try {
    const confirmRes = await fetch('/query/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId.value,
        pending_request_id: pendingRequestId.value,
        candidate_id: candidateName
      })
    })

    if (confirmRes.ok) {
      const confirmData = await confirmRes.json()
      if (confirmData.request_id) {
        newReqId = confirmData.request_id
      }
    }
  } catch (e) {
    console.warn('调用 confirm 接口异常，使用回退模型', e)
  }

  let isDone = false
  let pollAttempts = 0
  while (!isDone && pollAttempts < 300) {
    await new Promise(r => setTimeout(r, 500))
    pollAttempts++
    await pollTaskNodeStatus(newReqId)
    if (activeNodeSteps.value.every(n => n.status === 'completed')) {
      isDone = true
    }
  }

  isThinking.value = false
  
  // 从后端同步最新的完整会话记录（重试等待 MongoDB 异步落盘完成）
  let sessionDetail = await api.getSessionDetail(currentSessionId.value)
  let lastMsg = sessionDetail?.messages?.length ? sessionDetail.messages[sessionDetail.messages.length - 1] : null

  let retries = 0
  while ((!lastMsg || lastMsg.role !== 'assistant') && retries < 20) {
    await new Promise(r => setTimeout(r, 400))
    retries++
    sessionDetail = await api.getSessionDetail(currentSessionId.value)
    lastMsg = sessionDetail?.messages?.length ? sessionDetail.messages[sessionDetail.messages.length - 1] : null
  }

  if (sessionDetail && sessionDetail.messages && sessionDetail.messages.length > 0) {
    const lastMessage = sessionDetail.messages[sessionDetail.messages.length - 1]
    if (lastMessage.role === 'assistant') {
      messages.value = sessionDetail.messages
      messages.value[messages.value.length - 1].node_steps = JSON.parse(JSON.stringify(activeNodeSteps.value))
    }
  }
  scrollToBottom()
  await loadSessions()
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
  border-color: #6366f1;
  color: #6366f1;
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
