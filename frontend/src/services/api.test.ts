import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

const TASKS_STORAGE_KEY = 'rag_import_tasks'

describe('api.getTasks', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('移除后端已不存在的任务并停止后续轮询', async () => {
    localStorage.setItem(TASKS_STORAGE_KEY, JSON.stringify([{
      task_id: 'req-stale',
      file_name: 'stale.pdf',
      file_size: '1.00 MB',
      status: 'processing',
      created_at: '2026/08/17 00:00:00',
      nodes: [],
    }]))
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 404 }))
    vi.stubGlobal('fetch', fetchMock)

    expect(await api.getTasks()).toEqual([])
    expect(JSON.parse(localStorage.getItem(TASKS_STORAGE_KEY) || '[]')).toEqual([])

    await api.getTasks()
    expect(fetchMock).toHaveBeenCalledOnce()
  })
})
