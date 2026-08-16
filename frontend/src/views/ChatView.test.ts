import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '../services/api'
import { streamSse } from '../services/sse'
import ChatView from './ChatView.vue'

vi.mock('../services/api', () => ({
  api: {
    getSessions: vi.fn(),
    getSessionDetail: vi.fn(),
    sendQuery: vi.fn(),
    confirmQuery: vi.fn(),
    deleteSession: vi.fn(),
    clearAllSessions: vi.fn(),
    getTaskStatus: vi.fn(),
  },
}))

vi.mock('../services/sse', () => ({
  streamSse: vi.fn(),
}))

describe('ChatView 流式错误', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem('rag_is_stream', 'true')
    vi.mocked(api.getSessions).mockResolvedValue([])
    vi.mocked(api.sendQuery).mockResolvedValue({
      request_id: 'request-a',
      session_id: 'session-a',
      status: 'processing',
    })
    vi.mocked(streamSse).mockImplementation(async (_input, _init, onEvent) => {
      await onEvent({
        type: 'error',
        data: {
          code: 'embedding_unavailable',
          message: '向量生成服务暂时不可用',
          retryable: true,
        },
      })
    })
  })

  it('展示 error 终态原文并停止 loading', async () => {
    const wrapper = mount(ChatView)
    await flushPromises()

    await wrapper.get('textarea').setValue('什么是 RAG？')
    await wrapper.get('.send-btn').trigger('click')
    await flushPromises()

    expect(streamSse).toHaveBeenCalledWith('/stream/request-a', {}, expect.any(Function))
    expect(wrapper.text()).toContain('向量生成服务暂时不可用')
    expect(wrapper.find('.typewriter-cursor').exists()).toBe(false)
    await wrapper.get('textarea').setValue('继续提问')
    expect(wrapper.get('.send-btn').attributes('disabled')).toBeUndefined()
  })
})
