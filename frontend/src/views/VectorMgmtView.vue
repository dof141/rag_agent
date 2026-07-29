<template>
  <div class="vector-container">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon purple"><Cpu /></div>
        <div class="stat-info">
          <span class="stat-value">{{ items.length }}</span>
          <span class="stat-label">笔记分类数 (Notebook Topics)</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon blue"><Layers /></div>
        <div class="stat-info">
          <span class="stat-value">{{ totalChunks }}</span>
          <span class="stat-label">Milvus 切片向量总数 (Vector Chunks)</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon green"><Database /></div>
        <div class="stat-info">
          <span class="stat-value">BGE-M3 (1024D)</span>
          <span class="stat-label">Dense + Sparse 混合索引库</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon orange"><HardDrive /></div>
        <div class="stat-info">
          <span class="stat-value">MinIO Bucket</span>
          <span class="stat-label">多模态图片存储</span>
        </div>
      </div>
    </div>

    <div class="table-card">
      <div class="table-header-bar">
        <div class="search-box">
          <Search class="search-icon" />
          <input 
            v-model="searchKeyword" 
            placeholder="搜索笔记主题、资料标题或文件名..." 
            class="search-input"
          />
        </div>

        <button class="refresh-btn" @click="loadData">
          <RefreshCw class="refresh-icon" />
          <span>刷新数据</span>
        </button>
      </div>

      <div class="table-wrapper">
        <table class="custom-table">
          <thead>
            <tr>
              <th>笔记分类/学习主题 (Item Name)</th>
              <th>原始文件名 (File Title)</th>
              <th>切片数量 (Chunks)</th>
              <th>向量配置</th>
              <th>导入时间</th>
              <th>操作 (Actions)</th>
            </tr>
          </thead>

          <tbody>
            <tr v-for="item in filteredItems" :key="item.id">
              <td>
                <div class="item-cell">
                  <span class="item-name-badge">{{ item.item_name }}</span>
                </div>
              </td>

              <td>
                <span class="file-title-text">{{ item.file_title }}</span>
              </td>

              <td>
                <span class="chunk-badge">{{ item.chunk_count }} 条</span>
              </td>

              <td>
                <div class="vector-spec">
                  <span class="spec-tag">Dense 1024D</span>
                  <span class="spec-tag sparse" v-if="item.has_sparse">Sparse IP</span>
                </div>
              </td>

              <td>
                <span class="time-text">{{ item.created_at }}</span>
              </td>

              <td>
                <div class="action-cell">
                  <button class="action-btn view-btn" @click="openChunkDrawer(item)">
                    <Eye class="btn-icon" />
                    <span>查看切片</span>
                  </button>

                  <button class="action-btn del-btn" @click="confirmDelete(item)">
                    <Trash2 class="btn-icon" />
                    <span>删除向量</span>
                  </button>
                </div>
              </td>
            </tr>

            <tr v-if="filteredItems.length === 0">
              <td colspan="6" class="empty-cell">未检索到匹配的向量库数据</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <ChunkDetailDrawer 
      v-model="showDrawer"
      :chunk="selectedChunk"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Cpu, Layers, Database, HardDrive, Search, RefreshCw, Eye, Trash2 } from 'lucide-vue-next'
import type { KBItem, KBChunk } from '../types'
import { api } from '../services/api'
import ChunkDetailDrawer from '../components/ChunkDetailDrawer.vue'
import { ElMessageBox, ElMessage } from 'element-plus'

const items = ref<KBItem[]>([])
const searchKeyword = ref<string>('')

const showDrawer = ref<boolean>(false)
const selectedChunk = ref<KBChunk | null>(null)

onMounted(async () => {
  await loadData()
})

const loadData = async () => {
  items.value = await api.getKBItems()
}

const totalChunks = computed(() => {
  return items.value.reduce((acc, cur) => acc + cur.chunk_count, 0)
})

const filteredItems = computed(() => {
  const kw = searchKeyword.value.trim().toLowerCase()
  if (!kw) return items.value
  return items.value.filter(i => 
    i.item_name.toLowerCase().includes(kw) || 
    i.file_title.toLowerCase().includes(kw)
  )
})

const openChunkDrawer = async (item: KBItem) => {
  const chunks = await api.getKBChunks(item.item_name)
  if (chunks.length > 0) {
    selectedChunk.value = chunks[0]
  } else {
    selectedChunk.value = {
      chunk_id: 1001,
      file_title: item.file_title,
      item_name: item.item_name,
      title: '示例切片',
      parent_title: '文档根节点',
      part: 1,
      content: `设备 ${item.item_name} 的核心描述切片文本已成功索引于 Milvus 向量库中。`,
      dense_vector_preview: [0.012, -0.045, 0.089, 0.124],
      sparse_vector_preview: { 101: 0.95 }
    }
  }
  showDrawer.value = true
}

const confirmDelete = (item: KBItem) => {
  ElMessageBox.confirm(
    `确定要物理删除设备主体 [${item.item_name}] 及其关联的全部 Milvus 向量切片吗？删除后不可恢复！`,
    '⚠️ 知识库向量删除确认',
    {
      confirmButtonText: '确定物理删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    const res = await api.deleteKBItem(item.item_name)
    ElMessage.success(res.message)
    await loadData()
  }).catch(() => {})
}
</script>

<style scoped>
.vector-container {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 1300px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}

.stat-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.stat-icon.purple { background: linear-gradient(135deg, #a855f7, #9333ea); }
.stat-icon.blue { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.stat-icon.green { background: linear-gradient(135deg, #22c55e, #16a34a); }
.stat-icon.orange { background: linear-gradient(135deg, #f97316, #ea580c); }

.stat-info { display: flex; flex-direction: column; }
.stat-value { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }
.stat-label { font-size: 0.78rem; color: var(--text-muted); }

/* Table Section */
.table-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.table-header-bar {
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
  width: 320px;
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

.refresh-btn {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 0.85rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.refresh-btn:hover { border-color: #6366f1; color: #6366f1; }
.refresh-icon { width: 14px; height: 14px; }

.table-wrapper {
  overflow-x: auto;
}

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

.item-name-badge {
  background: rgba(168, 85, 247, 0.15);
  color: #c084fc;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 8px;
}

.file-title-text { color: var(--text-primary); }
.chunk-badge { font-weight: 600; color: #3b82f6; }

.vector-spec { display: flex; gap: 6px; }
.spec-tag {
  background: rgba(59, 130, 246, 0.12);
  color: #60a5fa;
  font-size: 0.72rem;
  padding: 2px 6px;
  border-radius: 4px;
}

.spec-tag.sparse {
  background: rgba(34, 197, 94, 0.12);
  color: #4ade80;
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
.btn-icon { width: 14px; height: 14px; }

.empty-cell { text-align: center; color: var(--text-muted); padding: 40px; }
</style>
