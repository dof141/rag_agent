import { marked } from 'marked'

const isImageUrl = (url: string): boolean => {
  if (!url) return false
  const lower = url.toLowerCase()
  return (
    /\.(jpg|jpeg|png|gif|webp|svg)(\?.*)?$/i.test(lower) ||
    lower.includes('/upload-images/') ||
    lower.includes('/images/')
  )
}

// 注册 marked 自定义渲染器：如果链接的目标是图片/MinIO存储图，直接将其渲染为 <img> 标签
marked.use({
  renderer: {
    link(arg1: any, _arg2?: any, arg3?: any) {
      const href = typeof arg1 === 'string' ? arg1 : (arg1?.href || '')
      const text = typeof arg3 === 'string' ? arg3 : (arg1?.text || arg1?.tokens?.[0]?.raw || href)

      if (isImageUrl(href)) {
        return `<img src="${href}" alt="${text || '设备图片'}" class="chat-inline-img" style="max-width:100%; border-radius:12px; margin:12px 0; display:block; border:1px solid var(--border-color, rgba(255,255,255,0.15)); box-shadow:0 6px 20px rgba(0,0,0,0.25);" />`
      }
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`
    },
    image(arg1: any) {
      const href = typeof arg1 === 'string' ? arg1 : (arg1?.href || '')
      const text = typeof arg1 === 'object' ? (arg1?.text || '设备图片') : '设备图片'
      return `<img src="${href}" alt="${text}" class="chat-inline-img" style="max-width:100%; border-radius:12px; margin:12px 0; display:block; border:1px solid var(--border-color, rgba(255,255,255,0.15)); box-shadow:0 6px 20px rgba(0,0,0,0.25);" />`
    }
  }
})

export const renderMarkdown = (text: string): string => {
  if (!text) return ''

  // 1. 将裸露的包含图片扩展名或 MinIO upload-images 路径的 URL 转为 [URL](URL) 格式，以便被 marked link 钩子捕获
  let processed = text.replace(
    /(^|[\s\n])(https?:\/\/[^\s\)\>\]]+?(?:\/upload-images\/|\/images\/|\.jpg|\.png|\.jpeg|\.webp|\.gif)[^\s\)\>\]]*)/gi,
    (_match, prefix, url) => {
      return `${prefix}[${url}](${url})`
    }
  )

  return marked.parse(processed) as string
}
