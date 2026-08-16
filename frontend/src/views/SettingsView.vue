<template>
  <div class="settings-page">
    <header class="settings-header">
      <div>
        <h2>运行配置</h2>
        <p>配置当前用户的向量生成与入库目标。</p>
      </div>
      <button class="save-button" data-test="save-settings" :disabled="loading" @click="save">
        <Save :size="16" />
        <span>{{ loading ? '保存中' : '保存配置' }}</span>
      </button>
    </header>

    <p v-if="message" class="message">{{ message }}</p>
    <p v-if="error" class="error">{{ error }}</p>

    <section class="settings-section">
      <h3>Embedding</h3>
      <div class="form-grid">
        <label>
          <span>Provider</span>
          <select v-model="form.embedding_provider">
            <option value="siliconflow">SiliconFlow</option>
            <option value="local_bge_m3">Local BGE-M3</option>
          </select>
        </label>
        <label>
          <span>Base URL</span>
          <input v-model="form.embedding_base_url" />
        </label>
        <label>
          <span>Model</span>
          <input v-model="form.embedding_model" />
        </label>
        <label>
          <span>Dimension</span>
          <input v-model.number="form.embedding_dimension" type="number" min="1" />
        </label>
        <label>
          <span>Batch Size</span>
          <input v-model.number="form.embedding_batch_size" type="number" min="1" max="128" />
        </label>
        <label>
          <span>Timeout</span>
          <input v-model.number="form.embedding_timeout" type="number" min="1" max="300" />
        </label>
      </div>
      <div class="secret-row">
        <label>
          <span>API Key</span>
          <input
            v-model="form.embedding_api_key"
            data-test="embedding-key"
            type="password"
            autocomplete="off"
            placeholder="留空表示保留旧值"
          />
        </label>
        <span class="masked">{{ settings?.embedding_api_key.masked || '未配置' }}</span>
        <button class="secondary-button" data-test="clear-embedding-key" @click="clearSecret('embedding_api_key')">
          清除
        </button>
      </div>
    </section>

    <section class="settings-section">
      <h3>向量库</h3>
      <div class="segmented">
        <button :class="{ active: form.vector_store_type === 'qdrant' }" @click="form.vector_store_type = 'qdrant'">
          Qdrant
        </button>
        <button :class="{ active: form.vector_store_type === 'milvus' }" @click="form.vector_store_type = 'milvus'">
          Milvus
        </button>
      </div>

      <p v-if="form.vector_store_type === 'qdrant'" class="notice">
        当前仅支持导入，查询迁移将在下一阶段完成
      </p>

      <div v-if="form.vector_store_type === 'qdrant'" class="form-grid">
        <label>
          <span>Qdrant URL</span>
          <input v-model="form.qdrant_url" />
        </label>
        <label>
          <span>Item Collection</span>
          <input v-model="form.qdrant_item_collection" />
        </label>
        <label>
          <span>Chunks Collection</span>
          <input v-model="form.qdrant_chunks_collection" />
        </label>
        <label class="checkbox-row">
          <input v-model="form.qdrant_cloud_inference" type="checkbox" />
          <span>启用 Qdrant Cloud Inference</span>
        </label>
      </div>

      <div v-if="form.vector_store_type === 'qdrant'" class="secret-row">
        <label>
          <span>Qdrant API Key</span>
          <input v-model="form.qdrant_api_key" type="password" autocomplete="off" placeholder="留空表示保留旧值" />
        </label>
        <span class="masked">{{ settings?.qdrant_api_key.masked || '未配置' }}</span>
        <button class="secondary-button" @click="clearSecret('qdrant_api_key')">清除</button>
      </div>

      <div v-if="form.vector_store_type === 'milvus'" class="form-grid">
        <label>
          <span>Milvus URL</span>
          <input v-model="form.milvus_url" />
        </label>
        <label>
          <span>Item Collection</span>
          <input v-model="form.milvus_item_collection" />
        </label>
        <label>
          <span>Chunks Collection</span>
          <input v-model="form.milvus_chunks_collection" />
        </label>
      </div>

      <div v-if="form.vector_store_type === 'milvus'" class="secret-row">
        <label>
          <span>Milvus Token</span>
          <input v-model="form.milvus_token" type="password" autocomplete="off" placeholder="留空表示保留旧值" />
        </label>
        <span class="masked">{{ settings?.milvus_token.masked || '未配置' }}</span>
        <button class="secondary-button" @click="clearSecret('milvus_token')">清除</button>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Save } from 'lucide-vue-next'
import { api, ApiError } from '../services/api'
import type { RuntimeSettingsResponse, RuntimeSettingsUpdate } from '../types'

const settings = ref<RuntimeSettingsResponse | null>(null)
const loading = ref(false)
const message = ref('')
const error = ref('')

const form = reactive<RuntimeSettingsUpdate>({
  embedding_provider: 'siliconflow',
  embedding_base_url: 'https://api.siliconflow.cn',
  embedding_model: 'BAAI/bge-m3',
  embedding_dimension: 1024,
  embedding_batch_size: 16,
  embedding_timeout: 30,
  embedding_api_key: '',
  vector_store_type: 'qdrant',
  qdrant_url: '',
  qdrant_api_key: '',
  qdrant_item_collection: 'rag_item_names_v1',
  qdrant_chunks_collection: 'rag_chunks_v1',
  qdrant_cloud_inference: true,
  milvus_url: '',
  milvus_token: '',
  milvus_item_collection: 'rag_item_names_v1',
  milvus_chunks_collection: 'rag_chunks_v1',
})

function applySettings(value: RuntimeSettingsResponse | null) {
  settings.value = value
  if (!value) return
  Object.assign(form, {
    embedding_provider: value.embedding_provider,
    embedding_base_url: value.embedding_base_url,
    embedding_model: value.embedding_model,
    embedding_dimension: value.embedding_dimension,
    embedding_batch_size: value.embedding_batch_size,
    embedding_timeout: value.embedding_timeout,
    embedding_api_key: '',
    vector_store_type: value.vector_store_type,
    qdrant_url: value.qdrant_url || '',
    qdrant_api_key: '',
    qdrant_item_collection: value.qdrant_item_collection || 'rag_item_names_v1',
    qdrant_chunks_collection: value.qdrant_chunks_collection || 'rag_chunks_v1',
    qdrant_cloud_inference: value.qdrant_cloud_inference,
    milvus_url: value.milvus_url || '',
    milvus_token: '',
    milvus_item_collection: value.milvus_item_collection || 'rag_item_names_v1',
    milvus_chunks_collection: value.milvus_chunks_collection || 'rag_chunks_v1',
  })
}

function showError(exc: unknown) {
  error.value = exc instanceof ApiError || exc instanceof Error ? exc.message : '操作失败'
  message.value = ''
}

onMounted(async () => {
  try {
    applySettings(await api.getRuntimeSettings())
  } catch (exc) {
    showError(exc)
  }
})

async function save() {
  loading.value = true
  error.value = ''
  try {
    applySettings(await api.saveRuntimeSettings({ ...form }))
    message.value = '配置已保存'
  } catch (exc) {
    showError(exc)
  } finally {
    loading.value = false
  }
}

async function clearSecret(name: 'embedding_api_key' | 'qdrant_api_key' | 'milvus_token') {
  if (!window.confirm('确认清除该密钥？')) return
  loading.value = true
  error.value = ''
  try {
    applySettings(await api.clearRuntimeSecret(name))
    message.value = '密钥已清除'
  } catch (exc) {
    showError(exc)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.settings-page {
  max-width: 1040px;
  margin: 0 auto;
  padding: 24px;
  display: grid;
  gap: 18px;
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.settings-header h2 {
  margin: 0;
  font-size: 24px;
  color: var(--text-primary);
}

.settings-header p {
  margin: 6px 0 0;
  color: var(--text-muted);
}

.settings-section {
  display: grid;
  gap: 16px;
  padding: 20px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-card);
}

.settings-section h3 {
  margin: 0;
  font-size: 16px;
  color: var(--text-primary);
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}

label {
  display: grid;
  gap: 6px;
  font-size: 13px;
  color: var(--text-secondary);
}

input,
select {
  height: 38px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 0 10px;
  background: var(--bg-card);
  color: var(--text-primary);
}

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.checkbox-row input {
  width: 16px;
  height: 16px;
}

.secret-row {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto auto;
  align-items: end;
  gap: 12px;
}

.masked {
  min-height: 38px;
  display: inline-flex;
  align-items: center;
  color: var(--text-muted);
  font-size: 13px;
}

.save-button,
.secondary-button,
.segmented button {
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-hover);
  color: var(--text-primary);
  cursor: pointer;
}

.save-button {
  height: 38px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0 14px;
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}

.secondary-button {
  height: 38px;
  padding: 0 12px;
}

.segmented {
  display: inline-flex;
  width: max-content;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
}

.segmented button {
  height: 34px;
  padding: 0 16px;
  border: 0;
  border-radius: 0;
}

.segmented button.active {
  background: #2563eb;
  color: #fff;
}

.notice,
.message,
.error {
  margin: 0;
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 13px;
}

.notice {
  background: rgba(37, 99, 235, 0.08);
  color: #1d4ed8;
}

.message {
  background: rgba(34, 197, 94, 0.12);
  color: #15803d;
}

.error {
  background: rgba(239, 68, 68, 0.1);
  color: #dc2626;
}
</style>
