<template>
  <main class="login-page">
    <form class="login-panel" @submit.prevent="submit">
      <h1>RAG Agent</h1>
      <label>
        <span>用户名</span>
        <input v-model="username" autocomplete="username" />
      </label>
      <label>
        <span>密码</span>
        <input v-model="password" type="password" autocomplete="current-password" />
      </label>
      <p v-if="auth.error" class="error">{{ auth.error }}</p>
      <button type="submit" :disabled="auth.loading">
        <LogIn :size="18" />
        <span>{{ auth.loading ? '登录中' : '登录' }}</span>
      </button>
    </form>
  </main>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { LogIn } from 'lucide-vue-next'
import { useAuthStore } from '../stores/authStore'

const username = ref('')
const password = ref('')
const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

async function submit() {
  await auth.login(username.value, password.value)
  const redirect = typeof route.query.redirect === 'string' && route.query.redirect.startsWith('/')
    ? route.query.redirect
    : '/import'
  await router.replace(redirect)
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: #f6f7fb;
}

.login-panel {
  width: min(360px, calc(100vw - 32px));
  display: grid;
  gap: 16px;
  padding: 28px;
  background: #fff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  box-shadow: 0 16px 40px rgb(15 23 42 / 10%);
}

h1 {
  margin: 0 0 4px;
  font-size: 24px;
}

label {
  display: grid;
  gap: 8px;
  color: #374151;
  font-size: 14px;
}

input {
  height: 40px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  padding: 0 12px;
  font-size: 14px;
}

button {
  height: 40px;
  border: 0;
  border-radius: 6px;
  background: #2563eb;
  color: #fff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 14px;
  cursor: pointer;
}

button:disabled {
  opacity: 0.7;
  cursor: wait;
}

.error {
  margin: 0;
  color: #dc2626;
  font-size: 13px;
}
</style>
