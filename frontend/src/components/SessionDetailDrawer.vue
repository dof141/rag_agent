<template>
  <el-drawer
    v-model="visible"
    title="📜 会话完整对谈纪录 (Chat Session Replay)"
    size="650px"
    direction="rtl"
  >
    <div v-if="session" class="session-replay">
      <div class="session-header-info">
        <h3 class="session-title">{{ session.title }}</h3>
        <div class="tags">
          <span class="tag-item" v-for="name in session.item_names" :key="name">
            🏷️ {{ name }}
          </span>
          <span class="time-tag">
            🕒 {{ new Date(session.last_ts * 1000).toLocaleString() }}
          </span>
        </div>
      </div>

      <div class="message-timeline">
        <div 
          v-for="msg in session.messages || []" 
          :key="msg.id" 
          class="msg-bubble-box"
          :class="msg.role"
        >
          <div class="msg-avatar">
            {{ msg.role === 'user' ? '👤' : '🤖' }}
          </div>
          <div class="msg-body">
            <div class="msg-role-name">
              {{ msg.role === 'user' ? '用户 Question' : 'RAG Assistant Answer' }}
            </div>
            <div class="msg-content markdown-body" v-html="renderMarkdown(msg.text)"></div>

            <div v-if="msg.sources && msg.sources.length" class="sources-box">
              <span class="source-title">📌 召回来源 (Sources):</span>
              <div v-for="s in msg.sources" :key="s.chunk_id" class="source-item">
                <span class="s-name">[{{ s.title }}]</span>
                <span class="s-score">Score: {{ ((s.score ?? 0.9) * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </el-drawer>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatSession } from '../types'
import { renderMarkdown } from '../utils/markdown'

const props = defineProps<{
  modelValue: boolean
  session: ChatSession | null
}>()

const emit = defineEmits(['update:modelValue'])

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})
</script>

<style scoped>
.session-replay {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.session-header-info {
  background: var(--bg-hover);
  padding: 14px 18px;
  border-radius: 12px;
  border: 1px solid var(--border-color);
}

.session-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.tags {
  display: flex;
  gap: 10px;
  align-items: center;
}

.tag-item {
  background: rgba(99, 102, 241, 0.15);
  color: #818cf8;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.78rem;
  font-weight: 600;
}

.time-tag {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.message-timeline {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 10px;
}

.msg-bubble-box {
  display: flex;
  gap: 12px;
}

.msg-bubble-box.user {
  flex-direction: row-reverse;
}

.msg-avatar {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
}

.msg-body {
  max-width: 80%;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 12px 16px;
}

.msg-bubble-box.user .msg-body {
  background: linear-gradient(135deg, #6366f1, #4f46e5);
  color: white;
  border: none;
}

.msg-role-name {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.msg-bubble-box.user .msg-role-name {
  color: rgba(255, 255, 255, 0.8);
}

.msg-content {
  font-size: 0.88rem;
  line-height: 1.6;
}

.msg-content :deep(img) {
  max-width: 100%;
  border-radius: 12px;
  margin: 10px 0;
  border: 1px solid var(--border-color);
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  display: block;
}

.sources-box {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border-color);
  font-size: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.source-title {
  color: var(--text-muted);
  font-weight: 600;
}

.source-item {
  display: flex;
  justify-content: space-between;
  color: #a855f7;
}
</style>
