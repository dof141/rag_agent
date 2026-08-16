import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError } from './api'
import { authFetch } from './http'
import { streamSse } from './sse'

vi.mock('./http', () => ({
  authFetch: vi.fn(),
}))

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

const streamResponse = (source: string, splitAt: number[]) => {
  const bytes = new TextEncoder().encode(source)
  let offset = 0
  const chunks = [...splitAt, bytes.length].map((end) => {
    const chunk = bytes.slice(offset, end)
    offset = end
    return chunk
  })
  return new Response(new ReadableStream({
    start(controller) {
      chunks.forEach(chunk => controller.enqueue(chunk))
      controller.close()
    },
  }), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

describe('streamSse', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('通过认证请求解析跨 UTF-8 与跨帧分块，并在 final 后停止', async () => {
    const source = [
      'event: delta\ndata: {"delta":"知识"}\n\n',
      'event: progress\ndata: {"running_list":["检索"]}\n\n',
      'event: warning\ndata: {"code":"rerank","message":"降级"}\n\n',
      'event: final\ndata: {"answer":"知识答案"}\n\n',
      'event: error\ndata: {"message":"不应分发"}\n\n',
    ].join('')
    const firstChineseByte = new TextEncoder().encode(source.slice(0, source.indexOf('知'))).length
    vi.mocked(authFetch).mockResolvedValue(streamResponse(source, [firstChineseByte + 1, firstChineseByte + 2, 51]))
    const events: Array<{ type: string; data: Record<string, unknown> }> = []

    await streamSse('/stream/request-a', {}, (event) => {
      events.push(event)
    })

    expect(authFetch).toHaveBeenCalledWith('/stream/request-a', {
      headers: { Accept: 'text/event-stream' },
    })
    expect(events.map(event => event.type)).toEqual(['delta', 'progress', 'warning', 'final'])
    expect(events[0].data.delta).toBe('知识')
    expect(events[3].data.answer).toBe('知识答案')
  })

  it.each([
    ['confirmation_required', { request_id: 'request-a', candidates: [{ id: 'topic-a' }] }],
    ['error', { code: 'embedding_unavailable', message: '向量生成服务暂时不可用' }],
  ])('将 %s 作为终态并忽略后续事件', async (type, data) => {
    const source = `event: ${type}\ndata: ${JSON.stringify(data)}\n\nevent: delta\ndata: {"delta":"late"}\n\n`
    vi.mocked(authFetch).mockResolvedValue(streamResponse(source, [7, 19]))
    const events: string[] = []

    await streamSse('/stream/request-a', {}, (event) => {
      events.push(event.type)
    })

    expect(events).toEqual([type])
  })
})

describe('受保护的问答和历史 API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('query、confirm 和 history 全部使用 authFetch', async () => {
    vi.mocked(authFetch)
      .mockResolvedValueOnce(jsonResponse({ request_id: 'r1', session_id: 's1', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ request_id: 'r2', session_id: 's1', status: 'processing' }))
      .mockResolvedValueOnce(jsonResponse({ code: 200, data: [] }))
      .mockResolvedValueOnce(jsonResponse({ session_id: 's1', items: [] }))
      .mockResolvedValueOnce(jsonResponse({ code: 200, deleted_count: 1 }))
      .mockResolvedValueOnce(jsonResponse({ code: 200, deleted_count: 2 }))

    await api.sendQuery('问题', 's1', true)
    await api.confirmQuery('s1', 'r1', 'candidate-1')
    await api.getSessions()
    await api.getSessionDetail('s1')
    await api.deleteSession('s1')
    await api.clearAllSessions()

    expect(vi.mocked(authFetch).mock.calls.map(call => String(call[0]))).toEqual([
      '/query',
      '/query/confirm',
      '/api/history/sessions',
      '/history/s1',
      '/history/s1',
      '/api/history/sessions',
    ])
  })

  it('后端错误按公开 message 抛出且不会返回仿真答案', async () => {
    vi.mocked(authFetch).mockResolvedValue(jsonResponse({
      code: 'query_configuration_invalid',
      message: '问答运行配置不存在或不完整',
      retryable: false,
    }, 409))

    await expect(api.sendQuery('问题', 's1', true)).rejects.toMatchObject({
      status: 409,
      message: '问答运行配置不存在或不完整',
    } satisfies Partial<ApiError>)
  })
})
