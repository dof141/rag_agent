import { describe, expect, it } from 'vitest'
import { clearAccessToken, setAccessToken } from '../services/http'
import { requiresAuthentication } from './index'

describe('router guard', () => {
  it('无 token 访问受保护页面时进入登录页', () => {
    clearAccessToken()
    expect(requiresAuthentication({ path: '/import', meta: { requiresAuth: true } })).toEqual({
      path: '/login',
      query: { redirect: '/import' },
    })
  })

  it('有 token 访问登录页时进入导入页', () => {
    setAccessToken('token')
    expect(requiresAuthentication({ path: '/login', meta: { public: true } })).toEqual('/import')
  })

  it('公开登录页无需 token', () => {
    clearAccessToken()
    expect(requiresAuthentication({ path: '/login', meta: { public: true } })).toBe(true)
  })
})
