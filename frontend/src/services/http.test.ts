import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AUTH_UNAUTHORIZED_EVENT, authFetch, setAccessToken } from './http'

describe('authFetch', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('附加 Bearer token 且不破坏 FormData header', async () => {
    setAccessToken('test-token')
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await authFetch('/upload', { method: 'POST', body: new FormData() })
    const headers = new Headers(fetchMock.mock.calls[0][1].headers)
    expect(headers.get('Authorization')).toBe('Bearer test-token')
    expect(headers.has('Content-Type')).toBe(false)
  })

  it('401 清除 token 并通知应用', async () => {
    setAccessToken('expired-token')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })))
    const listener = vi.fn()
    window.addEventListener(AUTH_UNAUTHORIZED_EVENT, listener)
    await authFetch('/status/task-a')
    expect(localStorage.getItem('rag_access_token')).toBeNull()
    expect(listener).toHaveBeenCalledOnce()
  })
})
