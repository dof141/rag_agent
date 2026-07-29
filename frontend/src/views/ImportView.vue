<template>
  <div class="import-container">
    <div class="upload-card">
      <div 
        class="dropzone"
        :class="{ 'drag-over': isDragOver }"
        @dragover.prevent="isDragOver = true"
        @dragleave.prevent="isDragOver = false"
        @drop.prevent="handleDrop"
      >
        <UploadCloud class="cloud-icon" />
        <h3>点击或拖拽 PDF / Word / PPT / MD / TXT 学习资料与笔记文档至此处上传</h3>
        <p class="dropzone-hint">
          系统将自动触发 MinerU 多模态解析全流程：格式深度提取 (PDF / Word / PPT / 扫描件) ➔ VLM 多模态图片描述与 MinIO 存储 ➔ 结构感知智能切片 ➔ 学习主题与实体提炼 ➔ BGE-M3 混合向量化 (Dense+Sparse) ➔ Milvus 知识库落盘
        </p>

        <input 
          type="file" 
          ref="fileInputRef" 
          multiple 
          accept=".pdf,.doc,.docx,.ppt,.pptx,.md,.txt,.png,.jpg,.jpeg" 
          class="hidden-input" 
          @change="handleFileSelect"
        />

        <button class="select-files-btn" @click="triggerFileSelect">
          <FolderPlus class="btn-icon" />
          <span>选择本地多格式学习资料文件</span>
        </button>
      </div>
    </div>

    <div class="tasks-section">
      <div class="section-header">
        <div class="header-left">
          <h3>📋 文档导入与节点监控 Task Pipeline</h3>
          <span class="task-count">共 {{ tasks.length }} 个解析任务</span>
        </div>
        <button v-if="tasks.length > 0" class="clear-tasks-btn" @click="clearTasks">
          <Trash2 class="btn-icon" />
          <span>清空任务记录</span>
        </button>
      </div>

      <div v-if="tasks.length === 0" class="empty-tasks-card">
        <FileText class="empty-icon" />
        <p>暂无正在运行或历史导入任务，请从上方拖拽上传 PDF / Word / PPT / MD / TXT 等多格式学习资料触发 MinerU 解析全流程</p>
      </div>

      <div v-else class="task-list">
        <div v-for="task in tasks" :key="task.task_id" class="task-card" :class="{ 'has-error': task.status === 'failed' }">
          <div class="task-card-header">
            <div class="file-meta">
              <FileText class="file-icon" />
              <span class="file-name">{{ task.file_name }}</span>
              <span class="file-size">({{ task.file_size }})</span>
              <span v-if="task.total_duration !== undefined && task.total_duration > 0" class="total-duration">
                ⏱️ 总耗时 {{ task.total_duration.toFixed(2) }}s
              </span>
            </div>

            <div class="header-right-actions">
              <button 
                v-if="task.status === 'failed'" 
                class="retry-action-btn" 
                @click="handleRetry(task.task_id)" 
                title="从第一个节点重新运行全流程"
              >
                <RotateCcw class="retry-icon" />
                <span>重新导入</span>
              </button>

              <div class="task-status-badge" :class="task.status">
                <span class="status-dot"></span>
                <span>{{ getStatusText(task.status) }}</span>
              </div>
            </div>
          </div>

          <!-- Failed Error Banner -->
          <div v-if="task.status === 'failed'" class="task-error-banner" :title="task.raw_error || task.error_msg">
            <AlertCircle class="error-banner-icon" />
            <span>节点解析异常：{{ task.error_msg || '后端节点处理失败，请检查文件格式或网络' }}</span>
          </div>

          <div class="pipeline-timeline">
            <div 
              v-for="(node, index) in task.nodes" 
              :key="node.node_id" 
              class="pipeline-step"
              :class="node.status"
              :title="node.error_msg || node.description"
            >
              <div class="step-icon-circle">
                <Check v-if="node.status === 'completed'" class="step-icon" />
                <AlertCircle v-else-if="node.status === 'failed'" class="step-icon error-icon" />
                <Loader2 v-else-if="node.status === 'running'" class="step-icon spin" />
                <span v-else class="step-num">{{ index + 1 }}</span>
              </div>

              <div class="step-label">
                <span class="node-name">
                  {{ node.name }}
                  <span v-if="node.duration !== undefined" class="node-duration-tag">{{ node.duration.toFixed(2) }}s</span>
                  <span v-else-if="node.status === 'running'" class="node-duration-tag running-tag">进行中</span>
                </span>
                <span class="node-desc" :class="{ 'error-desc': node.status === 'failed' }">{{ node.error_msg || node.description }}</span>
              </div>

              <div v-if="index < task.nodes.length - 1" class="step-line" :class="getLineClass(node, task.nodes[index + 1])"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { UploadCloud, FolderPlus, FileText, Check, Loader2, Trash2, AlertCircle, RotateCcw } from 'lucide-vue-next'
import type { ImportTask } from '../types'
import { api } from '../services/api'

const isDragOver = ref<boolean>(false)
const fileInputRef = ref<HTMLInputElement | null>(null)
const tasks = ref<ImportTask[]>([])
let pollTimer: any = null

onMounted(async () => {
  await fetchTasks()
  pollTimer = setInterval(async () => {
    await fetchTasks()
  }, 1500)
})

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer)
  }
})

const fetchTasks = async () => {
  tasks.value = await api.getTasks()
}

const triggerFileSelect = () => {
  fileInputRef.value?.click()
}

const handleFileSelect = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files) {
    uploadFiles(Array.from(target.files))
  }
}

const handleDrop = (e: DragEvent) => {
  isDragOver.value = false
  if (e.dataTransfer?.files) {
    uploadFiles(Array.from(e.dataTransfer.files))
  }
}

const uploadFiles = async (files: File[]) => {
  await api.uploadFiles(files)
  await fetchTasks()
}

const clearTasks = async () => {
  await api.clearTasks()
  tasks.value = []
}

const handleRetry = async (taskId: string) => {
  const ok = await api.retryTask(taskId)
  if (ok) {
    await fetchTasks()
  }
}

const getStatusText = (status: string) => {
  switch (status) {
    case 'completed': return '已完成入库'
    case 'processing': return '节点解析中...'
    case 'failed': return '处理失败'
    default: return '排队等待'
  }
}

const getLineClass = (curr: any, next: any) => {
  if (curr.status === 'completed' && next.status !== 'pending') return 'active'
  return ''
}
</script>

<style scoped>
.import-container {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.upload-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 30px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}

.dropzone {
  border: 2px dashed var(--border-color);
  border-radius: 12px;
  padding: 40px 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  transition: all 0.25s ease;
  background: var(--bg-hover);
}

.dropzone.drag-over {
  border-color: #6366f1;
  background: rgba(99, 102, 241, 0.08);
}

.cloud-icon {
  width: 48px;
  height: 48px;
  color: #6366f1;
}

.dropzone h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.dropzone-hint {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-align: center;
  max-width: 800px;
  line-height: 1.6;
}

.hidden-input {
  display: none;
}

.select-files-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.select-files-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
}

.btn-icon {
  width: 18px;
  height: 18px;
}

/* Tasks Pipeline */
.tasks-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-left h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.task-count {
  font-size: 0.8rem;
  color: var(--text-muted);
  background: var(--bg-hover);
  padding: 2px 10px;
  border-radius: 12px;
}

.clear-tasks-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.82rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-tasks-btn:hover {
  border-color: #ef4444;
  color: #ef4444;
  background: rgba(239, 68, 68, 0.08);
}

.empty-tasks-card {
  background: var(--bg-card);
  border: 1px dashed var(--border-color);
  border-radius: 12px;
  padding: 40px 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-muted);
  font-size: 0.88rem;
}

.empty-icon {
  width: 36px;
  height: 36px;
  opacity: 0.5;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.task-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.task-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.file-meta {
  display: flex;
  align-items: center;
  gap: 10px;
}

.file-icon { width: 20px; height: 20px; color: #a855f7; }
.file-name { font-weight: 600; font-size: 0.95rem; color: var(--text-primary); }
.file-size { font-size: 0.78rem; color: var(--text-muted); }

.total-duration {
  font-size: 0.75rem;
  font-weight: 600;
  color: #a855f7;
  background: rgba(168, 85, 247, 0.1);
  padding: 2px 8px;
  border-radius: 8px;
  border: 1px solid rgba(168, 85, 247, 0.2);
}

.node-duration-tag {
  font-size: 0.7rem;
  font-weight: 600;
  color: #22c55e;
  background: rgba(34, 197, 94, 0.12);
  padding: 1px 6px;
  border-radius: 6px;
  margin-left: 4px;
}

.node-duration-tag.running-tag {
  color: #3b82f6;
  background: rgba(59, 130, 246, 0.12);
}

.header-right-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.retry-action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
  color: #ffffff;
  border: none;
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25);
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.retry-action-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.45);
}

.retry-action-btn:hover .retry-icon {
  transform: rotate(-180deg);
}

.retry-icon {
  width: 13px;
  height: 13px;
  transition: transform 0.4s ease;
}

.task-status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
}

.task-status-badge.completed { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
.task-status-badge.processing { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
.task-status-badge.failed { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* Failed Error Banner */
.task-error-banner {
  margin: 6px 0 0 0;
  padding: 8px 12px;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 8px;
  color: #ef4444;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

.error-banner-icon {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

/* Timeline UI */
.pipeline-timeline {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  position: relative;
  padding: 10px 0;
}

.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
  text-align: center;
}

.step-icon-circle {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--bg-hover);
  border: 2px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 700;
  z-index: 2;
  transition: all 0.3s ease;
}

.pipeline-step.completed .step-icon-circle {
  background: #22c55e;
  border-color: #22c55e;
  color: white;
}

.pipeline-step.running .step-icon-circle {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
  box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
}

.pipeline-step.failed .step-icon-circle {
  background: #ef4444;
  border-color: #ef4444;
  color: white;
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
}

.step-icon { width: 16px; height: 16px; }
.step-icon.spin { animation: spin 1s linear infinite; }

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.step-label {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.node-name { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); }
.node-desc { font-size: 0.7rem; color: var(--text-muted); max-width: 110px; }
.node-desc.error-desc { color: #ef4444; font-weight: 600; }

.step-line {
  position: absolute;
  top: 16px;
  left: 50%;
  width: 100%;
  height: 2px;
  background: var(--border-color);
  z-index: 1;
}

.step-line.active {
  background: #22c55e;
}
</style>
