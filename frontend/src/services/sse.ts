import { authFetch } from './http'

export type SseEventType =
  | 'delta'
  | 'progress'
  | 'warning'
  | 'final'
  | 'confirmation_required'
  | 'error'

export interface SseEvent {
  type: SseEventType
  data: Record<string, any>
}

export type SseEventHandler = (event: SseEvent) => void | Promise<void>

const TERMINAL_EVENTS = new Set<SseEventType>([
  'final',
  'confirmation_required',
  'error',
])

const SUPPORTED_EVENTS = new Set<SseEventType>([
  'delta',
  'progress',
  'warning',
  ...TERMINAL_EVENTS,
])

export class SseRequestError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'SseRequestError'
    this.status = status
  }
}

const publicErrorMessage = async (response: Response): Promise<string> => {
  const data = await response.json().catch(() => null)
  if (typeof data?.message === 'string') return data.message
  if (typeof data?.detail === 'string') return data.detail
  return `请求失败：${response.status}`
}

const findFrameBoundary = (buffer: string): { index: number; length: number } | null => {
  const match = /\r?\n\r?\n/.exec(buffer)
  return match ? { index: match.index, length: match[0].length } : null
}

const parseFrame = (frame: string): SseEvent | null => {
  let type = 'message'
  const dataLines: string[] = []

  for (const rawLine of frame.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(':')) continue
    const separator = rawLine.indexOf(':')
    const field = separator === -1 ? rawLine : rawLine.slice(0, separator)
    let value = separator === -1 ? '' : rawLine.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') type = value
    if (field === 'data') dataLines.push(value)
  }

  if (!SUPPORTED_EVENTS.has(type as SseEventType) || dataLines.length === 0) return null
  let data: unknown
  try {
    data = JSON.parse(dataLines.join('\n'))
  } catch {
    throw new Error('SSE 事件数据格式错误')
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('SSE 事件数据格式错误')
  }
  return { type: type as SseEventType, data: data as Record<string, any> }
}

export async function streamSse(
  input: RequestInfo | URL,
  init: RequestInit = {},
  onEvent: SseEventHandler,
): Promise<void> {
  const headers = new Headers(init.headers)
  if (!headers.has('Accept')) headers.set('Accept', 'text/event-stream')
  const requestHeaders: HeadersInit = init.headers ? headers : { Accept: 'text/event-stream' }
  const response = await authFetch(input, { ...init, headers: requestHeaders })
  if (!response.ok) {
    throw new SseRequestError(response.status, await publicErrorMessage(response))
  }
  if (!response.body) throw new Error('浏览器无法读取问答流')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })

      let boundary = findFrameBoundary(buffer)
      while (boundary) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary.length)
        const event = parseFrame(frame)
        if (event) {
          await onEvent(event)
          if (TERMINAL_EVENTS.has(event.type)) {
            await reader.cancel()
            return
          }
        }
        boundary = findFrameBoundary(buffer)
      }

      if (done) return
    }
  } finally {
    reader.releaseLock()
  }
}
