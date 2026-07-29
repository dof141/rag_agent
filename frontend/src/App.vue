<template>
  <div class="app-layout">
    <Sidebar />
    <div class="main-content">
      <Header />
      <main class="router-viewport">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import Header from './components/Header.vue'
import { useAppStore } from './stores/appStore'

const store = useAppStore()

onMounted(async () => {
  document.documentElement.classList.remove('dark')
  await store.fetchStats()
})
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: var(--bg-main);
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.router-viewport {
  flex: 1;
  overflow-y: auto;
  position: relative;
}
</style>
