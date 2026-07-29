<template>
  <header class="header-bar">
    <div class="header-left">
      <h2 class="page-title">{{ route.meta.title || 'RAG Agent 管理平台' }}</h2>
    </div>

    <div class="header-right">
      <div class="model-badge">
        <span class="pulse-dot"></span>
        <span>LLM: Qwen / BGE-M3 (Dense+Sparse)</span>
      </div>

      <div class="user-info">
        <div class="avatar">Admin</div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()
</script>

<style scoped>
.header-bar {
  height: 64px;
  background: var(--bg-header);
  border-bottom: 1px solid var(--border-color);
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  backdrop-filter: blur(10px);
  z-index: 40;
}

.page-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text-primary);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.model-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.78rem;
  color: var(--text-secondary);
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #3b82f6;
  box-shadow: 0 0 8px #3b82f6;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { transform: scale(0.95); opacity: 0.8; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(0.95); opacity: 0.8; }
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: linear-gradient(135deg, #6366f1, #a855f7);
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}
</style>
