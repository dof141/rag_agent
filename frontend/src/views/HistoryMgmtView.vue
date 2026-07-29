<template>
  <div class="history-container">
    <div class="header-card">
      <div class="title-meta">
        <h3>📜 历史对话纪录管理 (MongoDB Chat History Management)</h3>
        <p>支持在线浏览历史 Session 概要、调取对话全量 Timeline 记录，以及单条或批量清除持久化数据。</p>
      </div>

      <button class="clear-all-btn" @click="confirmClearAll">
        <Trash2 class="btn-icon" />
        <span>清空全量历史纪录</span>
      </button>
    </div>

    <div class="sessions-card">
      <div class="filter-bar">
        <div class="search-box">
          <Search class="search-icon" />
          <input 
            v-model="keyword" 
            placeholder="按 Session ID、笔记主题或关键词搜索..." 
            class="search-input"
          />
        </div>
        <span class="count-tag">已显示 {{ filteredSessions.length }} 条 Session</span>
      </div>

      <div class="table-wrapper">
        <table class="custom-table">
          <thead>
            <tr>
              <th>Session ID</th>
              <th>对话主题 / 用户首提问</th>
              <th>消息条数</th>
              <th>关联笔记主题</th>
              <th>最后更新时间</th>
              <th>操作 (Actions)</th>
            </tr>
          </thead>

          <tbody>
            <tr v-for="s in filteredSessions" :key="s.session_id">
              <td>
                <span class="session-id-text font-mono">{{ s.session_id }}</span>
              </td>

              <td>
                <div class="title-cell">
                  <span class="session-title">{{ s.title }}</span>
                  <span class="session-snippet">{{ s.last_message }}</span>
                </div>
              </td>

              <td>
                <span class="count-badge">{{ s.message_count }} 条</span>
              </td>

              <td>
                <div class="tag-group">
                  <span class="item-tag" v-for="name in s.item_names" :key="name">{{ name }}</span>
                </div>
              </td>

              <td>
                <span class="time-text">{{ new Date(s.last_ts * 1000).toLocaleString() }}</span>
              </td>

              <td>
                <div class="action-cell">
                  <button class="action-btn view-btn" @click="replaySession(s)">
                    <Eye class="btn-icon" />
                    <span>查看对谈</span>
                  </button>

                  <button class="action-btn del-btn" @click="confirmDeleteOne(s)">
                    <Trash2 class="btn-icon" />
                    <span>删除记录</span>
                  </button>
                </div>
              </td>
            </tr>

            <tr v-if="filteredSessions.length === 0">
              <td colspan="6" class="empty-cell">未搜索到匹配的历史 Session 纪录</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <SessionDetailDrawer 
      v-model="showDrawer"
      :session="selectedSession"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Trash2, Search, Eye } from 'lucide-vue-next'
import type { ChatSession } from '../types'
import { api } from '../services/api'
import SessionDetailDrawer from '../components/SessionDetailDrawer.vue'
import { ElMessageBox, ElMessage } from 'element-plus'

const sessions = ref<ChatSession[]>([])
const keyword = ref<string>('')

const showDrawer = ref<boolean>(false)
const selectedSession = ref<ChatSession | null>(null)

onMounted(async () => {
  await loadSessions()
})

const loadSessions = async () => {
  sessions.value = await api.getSessions()
}

const filteredSessions = computed(() => {
  const kw = keyword.value.trim().toLowerCase()
  if (!kw) return sessions.value
  return sessions.value.filter(s => 
    s.session_id.toLowerCase().includes(kw) || 
    s.title.toLowerCase().includes(kw) ||
    s.item_names.some(n => n.toLowerCase().includes(kw))
  )
})

const replaySession = async (session: ChatSession) => {
  const detail = await api.getSessionDetail(session.session_id)
  selectedSession.value = detail || session
  showDrawer.value = true
}

const confirmDeleteOne = (session: ChatSession) => {
  ElMessageBox.confirm(
    `确定要删除 Session [${session.session_id}] 的全部历史对谈记录吗？`,
    '⚠️ 删除确认',
    {
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    await api.deleteSession(session.session_id)
    ElMessage.success('历史会话已成功删除')
    await loadSessions()
  }).catch(() => {})
}

const confirmClearAll = () => {
  ElMessageBox.confirm(
    '确定要物理清空 MongoDB 数据库中所有的历史对话纪录吗？此操作不可逆！',
    '🚨 警告：全量清空历史',
    {
      confirmButtonText: '确定一键清空',
      cancelButtonText: '取消',
      type: 'error'
    }
  ).then(async () => {
    const res = await api.clearAllSessions()
    ElMessage.success(`全量清空成功，已删除 ${res.deleted_count} 条记录`)
    await loadSessions()
  }).catch(() => {})
}
</script>

<style scoped>
.history-container {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 1300px;
  margin: 0 auto;
}

.header-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}

.title-meta h3 {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.title-meta p {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.clear-all-btn {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 10px;
  padding: 10px 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-all-btn:hover {
  background: #ef4444;
  color: white;
}

.btn-icon { width: 16px; height: 16px; }

.sessions-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-box {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 8px 14px;
  width: 360px;
}

.search-icon { width: 16px; height: 16px; color: var(--text-muted); }

.search-input {
  border: none;
  background: transparent;
  outline: none;
  color: var(--text-primary);
  font-size: 0.88rem;
  width: 100%;
}

.count-tag { font-size: 0.8rem; color: var(--text-muted); }

.table-wrapper { overflow-x: auto; }

.custom-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

.custom-table th {
  padding: 12px 16px;
  font-size: 0.8rem;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-color);
  font-weight: 600;
}

.custom-table td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-color);
  font-size: 0.88rem;
}

.session-id-text {
  color: #38bdf8;
  font-size: 0.8rem;
}

.title-cell { display: flex; flex-direction: column; gap: 2px; }
.session-title { font-weight: 600; color: var(--text-primary); }
.session-snippet { font-size: 0.78rem; color: var(--text-muted); }

.count-badge {
  background: rgba(59, 130, 246, 0.15);
  color: #60a5fa;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.78rem;
}

.tag-group { display: flex; gap: 6px; }
.item-tag {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 6px;
  font-weight: 600;
}

.time-text { color: var(--text-muted); font-size: 0.8rem; }

.action-cell { display: flex; gap: 8px; }

.action-btn {
  border: none;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: opacity 0.2s ease;
}

.view-btn { background: rgba(99, 102, 241, 0.15); color: #818cf8; }
.del-btn { background: rgba(239, 68, 68, 0.15); color: #f87171; }
.action-btn:hover { opacity: 0.8; }

.empty-cell { text-align: center; color: var(--text-muted); padding: 40px; }
</style>
