export const TOKEN_STORAGE_KEY = 'rag_access_token'
export const AUTH_UNAUTHORIZED_EVENT = 'rag:auth-unauthorized'

export const getAccessToken = (): string | null => localStorage.getItem(TOKEN_STORAGE_KEY)
export const setAccessToken = (token: string): void => localStorage.setItem(TOKEN_STORAGE_KEY, token)
export const clearAccessToken = (): void => localStorage.removeItem(TOKEN_STORAGE_KEY)

export async function authFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = getAccessToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(input, { ...init, headers })
  if (response.status === 401) {
    clearAccessToken()
    window.dispatchEvent(new Event(AUTH_UNAUTHORIZED_EVENT))
  }
  return response
}
