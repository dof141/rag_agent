<template>
  <el-drawer
    v-model="visible"
    title="🔍 向量切片数据详情 (Milvus Chunk Inspection)"
    size="600px"
    direction="rtl"
    class="chunk-drawer"
  >
    <div v-if="chunk" class="drawer-content">
      <div class="meta-section">
        <div class="meta-row">
          <span class="label">Chunk ID:</span>
          <span class="value font-mono">{{ chunk.chunk_id }}</span>
        </div>
        <div class="meta-row">
          <span class="label">笔记分类/主题 (Notebook Topic):</span>
          <span class="badge purple">{{ chunk.item_name }}</span>
        </div>
        <div class="meta-row">
          <span class="label">原始文件 (File Title):</span>
          <span class="value">{{ chunk.file_title }}</span>
        </div>
        <div class="meta-row">
          <span class="label">父级标题 (Parent Title):</span>
          <span class="value">{{ chunk.parent_title }}</span>
        </div>
        <div class="meta-row">
          <span class="label">当前标题 (Title):</span>
          <span class="value font-semibold">{{ chunk.title }}</span>
        </div>
      </div>

      <div class="content-section">
        <h4 class="section-title">📄 切片文本内容 (Content Text):</h4>
        <div class="content-box">
          <pre>{{ chunk.content }}</pre>
        </div>
      </div>

      <div class="vector-section">
        <h4 class="section-title">🧠 稠密向量 (Dense Vector 1024维 Preview):</h4>
        <div class="vector-box">
          <code>
            [{{ (chunk.dense_vector_preview || [0.012, -0.045, 0.089, 0.124, -0.003, 0.076, 0.312, -0.198]).join(', ') }}, ... (共 1024 维)]
          </code>
        </div>

        <h4 class="section-title mt-4">Sparse 稀疏向量 (Token-Weight Dictionary):</h4>
        <div class="vector-box">
          <code>
            {{ JSON.stringify(chunk.sparse_vector_preview || { "102": 0.84, "501": 0.62, "1204": 0.95 }) }}
          </code>
        </div>
      </div>
    </div>
  </el-drawer>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { KBChunk } from '../types'

const props = defineProps<{
  modelValue: boolean
  chunk: KBChunk | null
}>()

const emit = defineEmits(['update:modelValue'])

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})
</script>

<style scoped>
.drawer-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.meta-section {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.meta-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.85rem;
}

.label {
  color: var(--text-muted);
  width: 140px;
  flex-shrink: 0;
}

.value {
  color: var(--text-primary);
  word-break: break-all;
}

.badge {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
}

.section-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.content-box {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 14px;
  max-height: 240px;
  overflow-y: auto;
  font-size: 0.85rem;
  line-height: 1.6;
  color: var(--text-secondary);
}

.content-box pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: inherit;
}

.vector-box {
  background: #0f172a;
  color: #38bdf8;
  border-radius: 8px;
  padding: 12px;
  font-size: 0.78rem;
  word-break: break-all;
  border: 1px solid rgba(56, 189, 248, 0.2);
}

.mt-4 {
  margin-top: 14px;
}
</style>
