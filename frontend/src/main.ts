import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import './style.css'
import { AUTH_UNAUTHORIZED_EVENT } from './services/http'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

window.addEventListener(AUTH_UNAUTHORIZED_EVENT, () => {
  router.replace('/login')
})

app.mount('#app')
