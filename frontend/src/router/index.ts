import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import ImportView from '../views/ImportView.vue'
import VectorMgmtView from '../views/VectorMgmtView.vue'
import HistoryMgmtView from '../views/HistoryMgmtView.vue'
import LoginView from '../views/LoginView.vue'
import { getAccessToken } from '../services/http'

export const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    redirect: '/chat',
    meta: { requiresAuth: true }
  },
  {
    path: '/login',
    name: 'Login',
    component: LoginView,
    meta: { public: true }
  },
  {
    path: '/chat',
    name: 'Chat',
    component: ChatView,
    meta: { title: '智能问答与检索', requiresAuth: true }
  },
  {
    path: '/import',
    name: 'Import',
    component: ImportView,
    meta: { title: '文档解析与节点监控', requiresAuth: true }
  },
  {
    path: '/knowledge',
    name: 'Knowledge',
    component: VectorMgmtView,
    meta: { title: '向量知识库与设备管理', requiresAuth: true }
  },
  {
    path: '/history',
    name: 'History',
    component: HistoryMgmtView,
    meta: { title: '历史会话管理', requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export function requiresAuthentication(to: { path: string; meta?: Record<string, unknown> }) {
  const hasToken = Boolean(getAccessToken())
  if (to.meta?.requiresAuth && !hasToken) {
    return { path: '/login', query: { redirect: to.path } }
  }
  if (to.path === '/login' && hasToken) {
    return '/import'
  }
  return true
}

router.beforeEach((to) => requiresAuthentication(to))

export default router
