import { defineStore } from 'pinia'
import { clearAccessToken, setAccessToken } from '../services/http'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    loading: false,
    error: ''
  }),
  actions: {
    async login(username: string, password: string) {
      this.loading = true
      this.error = ''
      try {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        })
        if (!response.ok) {
          const data = await response.json().catch(() => ({}))
          throw new Error(data.detail || '用户名或密码错误')
        }
        const data = await response.json()
        setAccessToken(data.access_token)
      } catch (error) {
        this.error = error instanceof Error ? error.message : '登录失败'
        throw error
      } finally {
        this.loading = false
      }
    },
    logout() {
      clearAccessToken()
    }
  }
})
