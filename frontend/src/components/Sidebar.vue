<template>
  <aside 
    class="sidebar-container"
    :class="{ 'collapsed': store.isSidebarCollapsed }"
  >
    <!-- Brand / Logo -->
    <div class="logo-box">
      <div class="logo-icon">
        <Cpu class="icon-svg" />
      </div>
      <div v-if="!store.isSidebarCollapsed" class="logo-text">
        <span class="brand-name">RAG Agent</span>
        <span class="brand-tag">v0.2.0</span>
      </div>
    </div>

    <!-- Navigation Menu -->
    <nav class="nav-menu">
      <router-link 
        to="/chat" 
        class="nav-item"
        :class="{ 'active': route.path === '/chat' }"
      >
        <MessageSquare class="nav-icon" />
        <span v-if="!store.isSidebarCollapsed" class="nav-label">智能问答</span>
      </router-link>

      <router-link 
        to="/import" 
        class="nav-item"
        :class="{ 'active': route.path === '/import' }"
      >
        <UploadCloud class="nav-icon" />
        <span v-if="!store.isSidebarCollapsed" class="nav-label">文档解析与监控</span>
      </router-link>

      <router-link 
        to="/knowledge" 
        class="nav-item"
        :class="{ 'active': route.path === '/knowledge' }"
      >
        <Database class="nav-icon" />
        <span v-if="!store.isSidebarCollapsed" class="nav-label">向量知识库管理</span>
        <span v-if="!store.isSidebarCollapsed && store.stats" class="badge">
          {{ store.stats.total_items }}
        </span>
      </router-link>

      <router-link 
        to="/history" 
        class="nav-item"
        :class="{ 'active': route.path === '/history' }"
      >
        <Clock class="nav-icon" />
        <span v-if="!store.isSidebarCollapsed" class="nav-label">历史会话管理</span>
      </router-link>

      <router-link
        to="/settings"
        class="nav-item"
        :class="{ 'active': route.path === '/settings' }"
      >
        <Settings class="nav-icon" />
        <span v-if="!store.isSidebarCollapsed" class="nav-label">运行配置</span>
      </router-link>
    </nav>

    <!-- Bottom Status & Actions -->
    <div class="sidebar-footer">
      <div v-if="!store.isSidebarCollapsed" class="status-card">
        <div class="status-row">
          <span class="dot online"></span>
          <span class="status-text">运行配置已加载</span>
        </div>
        <div class="status-row">
          <span class="dot online"></span>
          <span class="status-text">上传链路待配置</span>
        </div>
      </div>

      <div class="footer-actions">
        <button class="action-btn" @click="store.toggleTheme" :title="store.isDark ? '切换浅色模式' : '切换深色模式'">
          <Sun v-if="store.isDark" class="action-icon" />
          <Moon v-else class="action-icon" />
        </button>

        <button class="action-btn" @click="store.toggleSidebar" title="折叠/展开侧边栏">
          <ChevronLeft v-if="!store.isSidebarCollapsed" class="action-icon" />
          <ChevronRight v-else class="action-icon" />
        </button>

        <button class="action-btn" @click="logout" title="退出登录">
          <LogOut class="action-icon" />
        </button>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '../stores/appStore'
import { useAuthStore } from '../stores/authStore'
import { 
  MessageSquare, 
  UploadCloud, 
  Database, 
  Clock, 
  Settings,
  Cpu, 
  Sun, 
  Moon, 
  ChevronLeft, 
  ChevronRight,
  LogOut
} from 'lucide-vue-next'

const route = useRoute()
const router = useRouter()
const store = useAppStore()
const auth = useAuthStore()

function logout() {
  auth.logout()
  router.replace('/login')
}
</script>

<style scoped>
.sidebar-container {
  width: 240px;
  height: 100vh;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  user-select: none;
  z-index: 50;
}

.sidebar-container.collapsed {
  width: 64px;
}

.logo-box {
  height: 64px;
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 12px;
  border-bottom: 1px solid var(--border-color);
}

.logo-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.icon-svg {
  width: 20px;
  height: 20px;
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.brand-name {
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}

.brand-tag {
  font-size: 0.7rem;
  color: #a855f7;
  font-weight: 600;
}

.nav-menu {
  flex: 1;
  padding: 16px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 10px 14px;
  border-radius: 10px;
  color: var(--text-secondary);
  text-decoration: none;
  font-weight: 500;
  font-size: 0.92rem;
  transition: all 0.2s ease;
  position: relative;
}

.nav-item:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.nav-item.active {
  background: var(--accent-gradient);
  color: white;
  box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25);
}

.nav-icon {
  width: 18px;
  height: 18px;
  margin-right: 12px;
  flex-shrink: 0;
}

.collapsed .nav-icon {
  margin-right: 0;
}

.collapsed .nav-item {
  justify-content: center;
  padding: 12px;
}

.badge {
  margin-left: auto;
  background: rgba(168, 85, 247, 0.2);
  color: #c084fc;
  padding: 2px 7px;
  border-radius: 12px;
  font-size: 0.72rem;
  font-weight: 600;
}

.sidebar-footer {
  padding: 14px;
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.dot.online {
  background: #22c55e;
  box-shadow: 0 0 6px #22c55e;
}

.footer-actions {
  display: flex;
  justify-content: space-between;
}

.action-btn {
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  border-radius: 8px;
  padding: 6px 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.action-btn:hover {
  color: var(--text-primary);
  background: var(--border-color);
}

.action-icon {
  width: 16px;
  height: 16px;
}
</style>
