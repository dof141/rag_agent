import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { SystemStats } from '../types'
import { api } from '../services/api'

export const useAppStore = defineStore('app', () => {
  const isDark = ref<boolean>(false)
  const stats = ref<SystemStats | null>(null)
  const isSidebarCollapsed = ref<boolean>(false)

  const toggleTheme = () => {
    isDark.value = !isDark.value
    if (isDark.value) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  const toggleSidebar = () => {
    isSidebarCollapsed.value = !isSidebarCollapsed.value
  }

  const fetchStats = async () => {
    stats.value = await api.getSystemStats()
  }

  return {
    isDark,
    stats,
    isSidebarCollapsed,
    toggleTheme,
    toggleSidebar,
    fetchStats
  }
})
