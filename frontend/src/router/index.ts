import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import ImportView from '../views/ImportView.vue'
import VectorMgmtView from '../views/VectorMgmtView.vue'
import HistoryMgmtView from '../views/HistoryMgmtView.vue'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    redirect: '/chat'
  },
  {
    path: '/chat',
    name: 'Chat',
    component: ChatView,
    meta: { title: '💬 智能问答与检索' }
  },
  {
    path: '/import',
    name: 'Import',
    component: ImportView,
    meta: { title: '📤 文档解析与节点监控' }
  },
  {
    path: '/knowledge',
    name: 'Knowledge',
    component: VectorMgmtView,
    meta: { title: '📚 向量知识库与设备管理' }
  },
  {
    path: '/history',
    name: 'History',
    component: HistoryMgmtView,
    meta: { title: '📜 历史会话管理' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
