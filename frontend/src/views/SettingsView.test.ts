import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError } from '../services/api'
import SettingsView from './SettingsView.vue'
import { authFetch } from '../services/http'
import type { RuntimeSettingsResponse, RuntimeSettingsUpdate } from '../types'

vi.mock('../services/http', () => ({
  authFetch: vi.fn(),
}))

const qdrantSettings: RuntimeSettingsResponse = {
  embedding_provider: 'siliconflow',
  embedding_base_url: 'https://api.siliconflow.cn',
  embedding_model: 'BAAI/bge-m3',
  embedding_dimension: 1024,
  embedding_batch_size: 16,
  embedding_timeout: 30,
  embedding_api_key: { configured: true, masked: 'sk-...flow' },
  vector_store_type: 'qdrant',
  qdrant_url: 'https://cluster.qdrant.io',
  qdrant_api_key: { configured: true, masked: 'qd-...key' },
  qdrant_item_collection: 'rag_item_names_v1',
  qdrant_chunks_collection: 'rag_chunks_v1',
  qdrant_cloud_inference: true,
  milvus_url: '',
  milvus_token: { configured: false, masked: null },
  milvus_item_collection: 'rag_item_names_v1',
  milvus_chunks_collection: 'rag_chunks_v1',
  version: 3,
  updated_at: '2026-08-16T08:00:00Z',
}

const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const qdrantPayload = (): RuntimeSettingsUpdate => ({
  embedding_provider: qdrantSettings.embedding_provider,
  embedding_base_url: qdrantSettings.embedding_base_url,
  embedding_model: qdrantSettings.embedding_model,
  embedding_dimension: qdrantSettings.embedding_dimension,
  embedding_batch_size: qdrantSettings.embedding_batch_size,
  embedding_timeout: qdrantSettings.embedding_timeout,
  embedding_api_key: '',
  vector_store_type: qdrantSettings.vector_store_type,
  qdrant_url: qdrantSettings.qdrant_url,
  qdrant_api_key: '',
  qdrant_item_collection: qdrantSettings.qdrant_item_collection,
  qdrant_chunks_collection: qdrantSettings.qdrant_chunks_collection,
  qdrant_cloud_inference: qdrantSettings.qdrant_cloud_inference,
  milvus_url: qdrantSettings.milvus_url,
  milvus_token: '',
  milvus_item_collection: qdrantSettings.milvus_item_collection,
  milvus_chunks_collection: qdrantSettings.milvus_chunks_collection,
})

describe('settings api', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('使用 authFetch 读取、保存和清除运行配置', async () => {
    vi.mocked(authFetch)
      .mockResolvedValueOnce(response(qdrantSettings))
      .mockResolvedValueOnce(response(qdrantSettings))
      .mockResolvedValueOnce(response(qdrantSettings))

    await expect(api.getRuntimeSettings()).resolves.toEqual(qdrantSettings)
    await api.saveRuntimeSettings(qdrantPayload())
    await api.clearRuntimeSecret('embedding_api_key')

    expect(authFetch).toHaveBeenNthCalledWith(1, '/api/settings/runtime')
    expect(authFetch).toHaveBeenNthCalledWith(
      2,
      '/api/settings/runtime',
      expect.objectContaining({ method: 'PUT' }),
    )
    expect(authFetch).toHaveBeenNthCalledWith(
      3,
      '/api/settings/runtime/secrets/embedding_api_key',
      { method: 'DELETE' },
    )
  })

  it('上传配置冲突时抛出稳定 ApiError', async () => {
    vi.mocked(authFetch).mockResolvedValue(response({ detail: 'SiliconFlow + Qdrant 配置不完整' }, 409))

    await expect(api.uploadFiles([new File(['hello'], 'hello.md')])).rejects.toMatchObject({
      status: 409,
      message: 'SiliconFlow + Qdrant 配置不完整',
    } satisfies Partial<ApiError>)
  })
})

describe('SettingsView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('空密钥保留旧值，清除必须单独确认', async () => {
    vi.mocked(authFetch)
      .mockResolvedValueOnce(response(qdrantSettings))
      .mockResolvedValueOnce(response(qdrantSettings))
      .mockResolvedValueOnce(response({ ...qdrantSettings, embedding_api_key: { configured: false, masked: null } }))

    const wrapper = mount(SettingsView)
    await flushPromises()

    const keyInput = wrapper.get<HTMLInputElement>('[data-test="embedding-key"]')
    expect(keyInput.element.value).toBe('')
    expect(wrapper.text()).toContain('sk-...flow')
    expect(wrapper.text()).toContain('当前仅支持导入，查询迁移将在下一阶段完成')

    await wrapper.get('[data-test="save-settings"]').trigger('click')
    await flushPromises()
    const saveCall = vi.mocked(authFetch).mock.calls[1]
    expect(JSON.parse(String(saveCall[1]?.body))).toEqual(
      expect.objectContaining({ embedding_api_key: '', qdrant_api_key: '' }),
    )

    vi.stubGlobal('confirm', vi.fn().mockReturnValue(true))
    await wrapper.get('[data-test="clear-embedding-key"]').trigger('click')
    await flushPromises()
    expect(authFetch).toHaveBeenLastCalledWith('/api/settings/runtime/secrets/embedding_api_key', {
      method: 'DELETE',
    })
  })
})
