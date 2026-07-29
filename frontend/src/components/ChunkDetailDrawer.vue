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
        <div class="section-header-bar">
          <h4 class="section-title">📄 切片文本内容 (Content Text):</h4>
          <div class="action-btns">
            <button v-if="!isEditing" class="action-mini-btn edit" @click="startEdit">
              ✏️ 编辑文本
            </button>

            <button v-if="isEditing" class="action-mini-btn save" :disabled="isSaving" @click="saveEdit">
              {{ isSaving ? '⏳ 重新向量化中...' : '💾 保存并重新向量化' }}
            </button>
            
            <button v-if="isEditing" class="action-mini-btn cancel" @click="cancelEdit">
              取消
            </button>

            <button class="action-mini-btn delete" @click="deleteChunk">
              🗑️ 删除切片
            </button>
          </div>
        </div>

        <div class="content-box">
          <textarea 
            v-if="isEditing" 
            v-model="editContent" 
            rows="6" 
            class="content-textarea" 
            placeholder="请输入修改后的切片文本..."
          ></textarea>
          <pre v-else>{{ chunk.content }}</pre>
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
import { ref, computed, watch } from 'vue'
import type { KBChunk } from '../types'
import { api } from '../services/api'
import { ElMessageBox, ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean
  chunk: KBChunk | null
}>()

const emit = defineEmits(['update:modelValue', 'chunkUpdated', 'chunkDeleted'])

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

const isEditing = ref(false)
const isSaving = ref(false)
const editContent = ref('')

watch(() => props.chunk, (newChunk) => {
  if (newChunk) {
    editContent.value = newChunk.content
    isEditing.value = false
  }
})

const startEdit = () => {
  if (props.chunk) {
    editContent.value = props.chunk.content
    isEditing.value = true
  }
}

const cancelEdit = () => {
  isEditing.value = false
}

const saveEdit = async () => {
  if (!props.chunk || !editContent.value.trim()) return
  isSaving.value = true
  try {
    const res = await api.updateKBChunk(props.chunk.chunk_id, editContent.value.trim())
    if (res.success) {
      ElMessage.success(res.message)
      props.chunk.content = editContent.value.trim()
      isEditing.value = false
      emit('chunkUpdated')
    } else {
      ElMessage.error(res.message || '更新失败')
    }
  } catch (e) {
    ElMessage.error('更新切片网络失败')
  } finally {
    isSaving.value = false
  }
}

const deleteChunk = () => {
  if (!props.chunk) return
  ElMessageBox.confirm(
    `确定要物理删除 Chunk ID 为 [${props.chunk.chunk_id}] 的这条切片吗？删除后 Milvus 数据库将移除对应的 1024 维向量！`,
    '⚠️ 单条切片物理删除确认',
    {
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    const res = await api.deleteKBChunk(props.chunk!.chunk_id)
    if (res.success) {
      ElMessage.success(res.message)
      visible.value = false
      emit('chunkDeleted')
    } else {
      ElMessage.error(res.message || '删除失败')
    }
  }).catch(() => {})
}
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

.section-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.section-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0;
}

.action-btns {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-mini-btn {
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 0.75rem;
  cursor: pointer;
  border: 1px solid var(--border-color);
  background: var(--bg-hover);
  color: var(--text-primary);
  transition: all 0.2s ease;
}

.action-mini-btn.edit {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  border-color: rgba(168, 85, 247, 0.3);
}

.action-mini-btn.save {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
  border-color: rgba(34, 197, 94, 0.3);
}

.action-mini-btn.cancel {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
}

.action-mini-btn.delete {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
  border-color: rgba(239, 68, 68, 0.3);
}

.content-textarea {
  width: 100%;
  background: var(--bg-main);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 10px;
  color: var(--text-primary);
  font-size: 0.85rem;
  line-height: 1.6;
  resize: vertical;
  outline: none;
}

.content-textarea:focus {
  border-color: #a855f7;
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
