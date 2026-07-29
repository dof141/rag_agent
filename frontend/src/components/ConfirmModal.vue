<template>
  <el-dialog
    v-model="visible"
    title="❓ 检测到多个匹配设备，请确认查询目标"
    width="520px"
    :close-on-click-modal="false"
    class="confirm-dialog"
  >
    <div class="confirm-body">
      <p class="desc">
        系统在中召回时识别到您的提问可能对应多个关联设备手册，请点击选择您想查询的具体设备：
      </p>

      <div class="candidate-list">
        <div 
          v-for="item in candidates" 
          :key="item.id" 
          class="candidate-card"
          @click="selectCandidate(item.item_name)"
        >
          <div class="card-left">
            <Cpu class="card-icon" />
            <div class="card-info">
              <span class="item-name">{{ item.item_name }}</span>
              <span class="file-name" v-if="item.file_title">{{ item.file_title }}</span>
            </div>
          </div>
          <div class="card-right" v-if="item.score">
            <span class="score-tag">匹配度 {{ (item.score * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { CandidateItem } from '../types'
import { Cpu } from 'lucide-vue-next'

const props = defineProps<{
  modelValue: boolean
  candidates: CandidateItem[]
}>()

const emit = defineEmits(['update:modelValue', 'confirm'])

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val)
})

const selectCandidate = (itemName: string) => {
  emit('confirm', itemName)
  visible.value = false
}
</script>

<style scoped>
.confirm-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.desc {
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.candidate-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.candidate-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-radius: 12px;
  background: var(--bg-hover);
  border: 1px solid var(--border-color);
  cursor: pointer;
  transition: all 0.25s ease;
}

.candidate-card:hover {
  border-color: #6366f1;
  background: rgba(99, 102, 241, 0.1);
  transform: translateY(-2px);
}

.card-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.card-icon {
  width: 22px;
  height: 22px;
  color: #6366f1;
}

.card-info {
  display: flex;
  flex-direction: column;
}

.item-name {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text-primary);
}

.file-name {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.score-tag {
  background: rgba(34, 197, 94, 0.15);
  color: #4ade80;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 6px;
}
</style>
