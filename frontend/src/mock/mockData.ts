import type { KBItem, KBChunk, ChatSession, ImportTask, SystemStats } from '../types'

export const mockStats: SystemStats = {
  total_items: 8,
  total_chunks: 142,
  total_sessions: 15,
  milvus_status: 'online',
  minio_status: 'online',
  mongo_status: 'online'
}

export const mockKBItems: KBItem[] = [
  {
    id: '1',
    item_name: 'hak180',
    file_title: 'hak180产品安全手册.pdf',
    chunk_count: 24,
    created_at: '2026-07-25 10:30:12',
    file_size: '1.24 MB',
    dense_dim: 1024,
    has_sparse: true
  },
  {
    id: '2',
    item_name: '万用表RS-12',
    file_title: '万用表RS-12的使用.pdf',
    chunk_count: 18,
    created_at: '2026-07-24 16:45:00',
    file_size: '1.35 MB',
    dense_dim: 1024,
    has_sparse: true
  },
  {
    id: '3',
    item_name: 'H3C ER2100',
    file_title: 'H3C ER2100企业级路由器 用户手册-6W104-整本手册.pdf',
    chunk_count: 36,
    created_at: '2026-07-23 11:20:05',
    file_size: '2.72 MB',
    dense_dim: 1024,
    has_sparse: true
  },
  {
    id: '4',
    item_name: 'HUAWEI MateBook B3-410',
    file_title: 'HUAWEI MateBook B3-410&B3-510 用户手册-(03,zh-cn,Boh&Nbl,HUAWEI).pdf',
    chunk_count: 29,
    created_at: '2026-07-22 09:15:30',
    file_size: '6.18 MB',
    dense_dim: 1024,
    has_sparse: true
  },
  {
    id: '5',
    item_name: '华为擎云 B530',
    file_title: '华为擎云B530 用户指南-(PUCZ,Windows11_03,zh-cn).pdf',
    chunk_count: 15,
    created_at: '2026-07-21 14:02:18',
    file_size: '670 KB',
    dense_dim: 1024,
    has_sparse: true
  },
  {
    id: '6',
    item_name: 'Pantum P3030',
    file_title: 'Pantum P3030 User Guide zh_CN V1_2.pdf',
    chunk_count: 20,
    created_at: '2026-07-20 17:50:44',
    file_size: '6.77 MB',
    dense_dim: 1024,
    has_sparse: true
  }
]

export const mockKBChunks: Record<string, KBChunk[]> = {
  'hak180': [
    {
      chunk_id: 467602001275654313,
      file_title: 'hak180产品安全手册.pdf',
      item_name: 'hak180',
      title: '## 设备安全与操作说明',
      parent_title: '# 1. 设备说明',
      part: 1,
      content: `## 设备安全与操作说明\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术人员进行全面工作以将本设备恢复到正常操作。\n\nBrother 不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n\n![hak180操作面板](http://127.0.0.1:9000/rag-agent-bucket/images/hak180/panel_diag.jpg)`,
      dense_vector_preview: [0.012, -0.045, 0.089, 0.124, -0.003, 0.076, 0.312, -0.198],
      sparse_vector_preview: { 102: 0.84, 501: 0.62, 1204: 0.95 }
    },
    {
      chunk_id: 467602001275654314,
      file_title: 'hak180产品安全手册.pdf',
      item_name: 'hak180',
      title: '## 烫印区域局部设置 (50mm - 170mm)',
      parent_title: '# 2. 高级参数设置',
      part: 2,
      content: `## 烫印区域局部设置 (50mm - 170mm)\n\n若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，请按以下步骤操作：\n\n1. 按下操作面板上的 **[Mode]** 键进入参数模式。\n2. 使用方向键切换至 **"Top Offset"** 设定项，将其修改为 50mm。\n3. 切换至 **"Print Length"** 设定项，将其修改为 120mm (即 170mm - 50mm)。\n4. 按下 **[OK]** 确认保存即可生效。`,
      dense_vector_preview: [0.034, 0.112, -0.088, 0.201, 0.054, -0.120, 0.045, 0.290],
      sparse_vector_preview: { 88: 0.91, 342: 0.77, 890: 0.85 }
    }
  ],
  '万用表RS-12': [
    {
      chunk_id: 467602001275654401,
      file_title: '万用表RS-12的使用.pdf',
      item_name: '万用表RS-12',
      title: '## 电池更换与保养规程',
      parent_title: '# 维护保养',
      part: 1,
      content: `## 电池更换与保养规程\n\n当万用表屏幕右上角显示低电量图标 🔋 时，应及时更换电池，避免测量数据偏差：\n\n1. 关闭万用表电源开关，将红黑表笔从测试插孔拔出。\n2. 拧开后盖板上的 2 颗固定螺丝。\n3. 取出旧 9V 叠层电池，按正负极标记装入新电池。\n4. 合上后盖板并拧紧螺丝。`,
      dense_vector_preview: [-0.012, 0.098, 0.154, -0.032, 0.088, 0.210, -0.045, 0.112],
      sparse_vector_preview: { 15: 0.89, 789: 0.94 }
    }
  ]
}

export const mockSessions: ChatSession[] = [
  {
    session_id: 'sess-hak180-001',
    title: 'HAK180 烫金区域设置咨询',
    last_message: '若想在纸张上只把烫金膜转印到顶部 50mm-170mm 的局部区域，请按以下步骤在操作面板上设置...',
    last_role: 'assistant',
    last_ts: 1785002400,
    message_count: 4,
    item_names: ['hak180'],
    messages: [
      {
        id: 'm1',
        role: 'user',
        text: 'HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置？',
        timestamp: 1785002300
      },
      {
        id: 'm2',
        role: 'assistant',
        text: '针对 **HAK 180 烫金机** 顶部 50 mm – 170 mm 的局部区域转印需求，设置步骤如下：\n\n1. **进入模式**：按操作面板上的 **[Mode]** 按钮进入设置选项。\n2. **设置起始偏移**：找到 `Top Offset` 选项，调整数值为 **50 mm**。\n3. **设置转印长度**：找到 `Print Length` 选项，将其设置为 **120 mm**（即 170 mm - 50 mm = 120 mm）。\n4. **保存设置**：按 **[OK]** 保存即可生效。\n\n【相关文档来源】：*hak180产品安全手册.pdf (第 2 章)*',
        item_names: ['hak180'],
        timestamp: 1785002400,
        sources: [
          {
            chunk_id: 467602001275654314,
            title: '烫印区域局部设置 (50mm - 170mm)',
            parent_title: '# 2. 高级参数设置',
            content: '按下操作面板上的 [Mode] 键进入参数模式。使用方向键切换至 "Top Offset" 设定项，将其修改为 50mm...',
            score: 0.942,
            source: 'local'
          }
        ]
      }
    ]
  },
  {
    session_id: 'sess-multimeter-002',
    title: 'RS-12 万用表更换电池问题',
    last_message: '万用表低电量提示时如何更换9V叠层电池？',
    last_role: 'user',
    last_ts: 1784998000,
    message_count: 2,
    item_names: ['万用表RS-12'],
    messages: [
      {
        id: 'm3',
        role: 'user',
        text: 'RS-12 万用表更换电池步骤？',
        timestamp: 1784997900
      },
      {
        id: 'm4',
        role: 'assistant',
        text: '当屏幕显示电池电量不足提示时，更换步骤为：\n1. 拔出红黑表笔并关机；\n2. 拧开后盖螺丝；\n3. 更换 9V 叠层电池。',
        item_names: ['万用表RS-12'],
        timestamp: 1784998000
      }
    ]
  }
]

export const mockImportTasks: ImportTask[] = [
  {
    task_id: 'task-upload-9981',
    file_name: 'H3C NER214W路由器 用户手册-6W101-整本手册.pdf',
    file_size: '5.74 MB',
    status: 'processing',
    created_at: '2026-07-26 14:05:00',
    nodes: [
      { node_id: 'node_entry', name: '入口校验', description: '参数非空校验与文件格式校验', status: 'completed', updated_at: '14:05:01' },
      { node_id: 'node_pdf_to_md', name: 'PDF转Markdown', description: 'MinerU 大模型多模态解析 PDF 文件', status: 'completed', updated_at: '14:05:15' },
      { node_id: 'node_md_img', name: 'MD图片处理', description: 'VLM 描述图片并上传 MinIO 对象存储', status: 'running', updated_at: '14:05:22' },
      { node_id: 'node_document_split', name: '文档智能切片', description: '基于标题结构二次切割与合并', status: 'pending' },
      { node_id: 'node_item_name_recognition', name: '设备主体识别', description: 'LLM 识别产品名称与索引构建', status: 'pending' },
      { node_id: 'node_bge_embedding', name: 'BGE-M3 向量化', description: '生成 1024 维稠密与稀疏 Token 向量', status: 'pending' },
      { node_id: 'node_import_milvus', name: 'Milvus 向量入库', description: '幂等清理与批量数据落盘', status: 'pending' }
    ]
  },
  {
    task_id: 'task-upload-9980',
    file_name: 'HUAWEI MateBook B5-420 用户手册.pdf',
    file_size: '5.70 MB',
    status: 'completed',
    created_at: '2026-07-26 13:40:00',
    nodes: [
      { node_id: 'node_entry', name: '入口校验', description: '校验通过', status: 'completed' },
      { node_id: 'node_pdf_to_md', name: 'PDF转Markdown', description: '解析成功 (Full.md)', status: 'completed' },
      { node_id: 'node_md_img', name: 'MD图片处理', description: '上传 12 张图片至 MinIO', status: 'completed' },
      { node_id: 'node_document_split', name: '文档智能切片', description: '成功切分 28 个智能 Chunk', status: 'completed' },
      { node_id: 'node_item_name_recognition', name: '设备主体识别', description: '识别主体: HUAWEI MateBook B5-420', status: 'completed' },
      { node_id: 'node_bge_embedding', name: 'BGE-M3 向量化', description: '混合向量生成完毕', status: 'completed' },
      { node_id: 'node_import_milvus', name: 'Milvus 向量入库', description: '成功入库 28 条数据', status: 'completed' }
    ]
  }
]
