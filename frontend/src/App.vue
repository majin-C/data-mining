<template>
  <main class="shell">
    <header class="topbar">
      <section>
        <h1>论文智能聚类与推荐系统</h1>
        <p>基于文本 Embedding 向量检索的 arXiv 论文分析</p>
      </section>
      <section class="actions">
        <button class="secondary" @click="loadAll" :disabled="loading">刷新数据</button>
        <button class="primary" @click="startIncrementalUpdate" :disabled="syncing">更新数据</button>
      </section>
    </header>

    <section class="stats">
      <article>
        <span>论文总数</span>
        <strong>{{ stats.paper_count ?? 0 }}</strong>
      </article>
      <article>
        <span>已生成向量</span>
        <strong>{{ stats.embedding_count ?? 0 }}</strong>
      </article>
    </section>

    <section class="workspace">
      <aside class="panel search-panel">
        <div class="control-row">
          <input v-model="searchQuery" placeholder="搜索标题或摘要" @keyup.enter="searchPapers" />
          <button @click="searchPapers">搜索</button>
        </div>
        <div class="control-row">
          <select v-model="category" @change="searchPapers">
            <option value="">全部分类</option>
            <option v-for="item in categories" :key="item.id" :value="item.id">
              {{ item.id }}（{{ item.count }}）
            </option>
          </select>
        </div>
        <div class="date-filter">
          <select v-model="timeField" @change="searchPapers">
            <option value="update_date">按更新日期</option>
            <option value="created_date">按创建日期</option>
          </select>
          <div class="date-inputs">
            <input v-model="dateFrom" type="date" @change="searchPapers" />
            <input v-model="dateTo" type="date" @change="searchPapers" />
          </div>
          <button class="secondary" @click="clearDateFilter" :disabled="!dateFrom && !dateTo">清空时间</button>
        </div>

        <div class="query-box">
          <textarea v-model="queryText" placeholder="输入研究兴趣，例如：large language model retrieval"></textarea>
          <button @click="recommendByQuery" :disabled="!queryText.trim()">按研究兴趣推荐</button>
        </div>

        <p class="status">{{ statusText }}</p>
        <div v-if="syncProgress.status !== 'idle'" class="sync-progress">
          <div class="sync-progress-track">
            <div class="sync-progress-bar" :style="{ width: `${syncProgressPercent}%` }"></div>
          </div>
          <span>{{ syncProgressLabel }}</span>
        </div>
      </aside>

      <section class="panel paper-list">
        <div class="section-title">
          <h2>论文列表</h2>
          <span>{{ total }} 篇结果</span>
        </div>
        <button
          v-for="paper in papers"
          :key="paper.id"
          class="paper-row"
          :class="{ active: selectedPaper && selectedPaper.id === paper.id }"
          @click="selectPaper(paper.id)"
        >
          <strong>{{ paper.title }}</strong>
          <span>{{ paper.categories }}</span>
          <small>{{ paper.abstract }}</small>
        </button>
      </section>

      <section class="panel detail-panel">
        <div class="section-title">
          <h2>论文详情</h2>
          <span v-if="selectedPaper">ID: {{ selectedPaper.id }}</span>
        </div>
        <div v-if="selectedPaper" class="detail">
          <h3>{{ selectedPaper.title }}</h3>
          <p class="meta">{{ selectedPaper.authors }}</p>
          <p class="badge">{{ selectedPaper.categories }}</p>
          <p>{{ selectedPaper.abstract }}</p>
          <button @click="recommendByPaper(selectedPaper.id)">推荐相似论文</button>
        </div>
        <p v-else class="empty">请选择一篇论文查看详情。</p>

        <div class="section-title recommendations-title">
          <h2>推荐结果</h2>
          <span>{{ recommendations.length }} 条</span>
        </div>
        <div class="recommendation-list">
          <article v-for="item in recommendations" :key="item.id" class="recommendation">
            <strong>{{ item.title }}</strong>
            <span>相似度 {{ Number(item.score || 0).toFixed(3) }}</span>
            <p>{{ item.abstract }}</p>
          </article>
        </div>
      </section>
    </section>

    <section class="panel k-panel">
      <div class="section-title">
        <h2>K 值评估</h2>
        <span>
          {{ kEvaluation.cached ? '本地缓存' : '实验结果' }} ·
          {{ kEvaluation.actual_sample_size || 0 }} / {{ kEvaluationDisplayTotal }} 条
        </span>
      </div>
      <div class="k-actions">
        <button class="secondary" @click="evaluateK(false)" :disabled="evaluatingK">评估 K 值</button>
        <button class="secondary" @click="confirmReevaluateK" :disabled="evaluatingK">刷新评估</button>
        <button class="primary" @click="applySelectedK" :disabled="evaluatingK || applyingK">
          应用全部选中的 K
        </button>
      </div>
      <div v-if="kProgress.status !== 'idle'" class="k-progress">
        <div class="k-progress-track">
          <div class="k-progress-bar" :style="{ width: `${kProgressPercent}%` }"></div>
        </div>
        <span>{{ kProgressLabel }}</span>
      </div>

      <div class="category-chart-grid">
        <article v-for="group in kEvaluationGroups" :key="group.id" class="category-chart-card">
          <div class="category-chart-title">
            <strong>{{ group.id }}</strong>
            <span>{{ group.paper_count || 0 }} 篇 · 当前 K={{ selectedKs[group.id] || defaultK }}</span>
          </div>
          <div :ref="(el) => setKChartRef(group.id, el)" class="chart k-chart"></div>
        </article>
      </div>
    </section>

    <section class="panel chart-panel">
      <div class="section-title">
        <h2>二维聚类可视化</h2>
        <span>8 个主分类分别展示，颜色表示分类内部子聚类</span>
      </div>
      <div class="cluster-chart-grid">
        <article v-for="group in clusterGroupsForDisplay" :key="group.id" class="cluster-chart-card">
          <div class="category-chart-title">
            <strong>{{ group.id }}</strong>
            <span>K={{ group.k || selectedKs[group.id] || defaultK }} · 展示 {{ group.shown_count || 0 }} / {{ group.paper_count || 0 }} 点</span>
          </div>
          <div :ref="(el) => setClusterChartRef(group.id, el)" class="chart cluster-chart"></div>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup>
import axios from 'axios'
import * as echarts from 'echarts'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const defaultCategoryOrder = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.IR', 'cs.DB', 'cs.SE', 'cs.DS']
const defaultK = 10
const initialSelectedKs = Object.fromEntries(defaultCategoryOrder.map((item) => [item, defaultK]))

const papers = ref([])
const total = ref(0)
const stats = ref({})
const categories = ref([])
const selectedPaper = ref(null)
const recommendations = ref([])
const searchQuery = ref('')
const category = ref('')
const timeField = ref('update_date')
const dateFrom = ref('')
const dateTo = ref('')
const queryText = ref('')
const statusText = ref('系统已就绪')
const loading = ref(false)
const syncing = ref(false)
const evaluatingK = ref(false)
const applyingK = ref(false)
const selectedKs = ref({ ...initialSelectedKs })
const kEvaluation = ref({
  cached: false,
  requested_sample_size: 0,
  actual_sample_size: 0,
  categories: defaultCategoryOrder.map((id) => ({ id, paper_count: 0, items: [] }))
})
const kProgress = ref({
  status: 'idle',
  completed: 0,
  total: defaultCategoryOrder.length * 12,
  current_k: null,
  current_category: null,
  error: null
})
const syncProgress = ref({
  status: 'idle',
  stage: 'idle',
  fetched: 0,
  inserted: 0,
  skipped: 0,
  embedded: 0,
  embedding_completed: 0,
  embedding_total: 0,
  message: '',
  error: null
})
const clusterGroups = ref(defaultCategoryOrder.map((id) => ({ id, paper_count: 0, shown_count: 0, points: [] })))

const kChartElements = new Map()
const clusterChartElements = new Map()
const kCharts = new Map()
const clusterCharts = new Map()
let kProgressTimer = null
let syncProgressTimer = null
let clickedButtonTimer = null

const knownCategoryOrder = computed(() => {
  const ids = new Set(defaultCategoryOrder)
  for (const item of categories.value) ids.add(item.id)
  for (const item of kEvaluation.value.categories || []) ids.add(item.id)
  for (const item of clusterGroups.value || []) ids.add(item.id)
  return [...defaultCategoryOrder.filter((item) => ids.has(item)), ...[...ids].filter((item) => !defaultCategoryOrder.includes(item)).sort()]
})

const kEvaluationGroups = computed(() => {
  const byId = new Map((kEvaluation.value.categories || []).map((item) => [item.id, item]))
  return knownCategoryOrder.value.map((id) => byId.get(id) || { id, paper_count: categoryCount(id), items: [] })
})

const clusterGroupsForDisplay = computed(() => {
  const byId = new Map((clusterGroups.value || []).map((item) => [item.id, item]))
  return knownCategoryOrder.value.map((id) => byId.get(id) || { id, paper_count: categoryCount(id), shown_count: 0, points: [] })
})

const kProgressPercent = computed(() => {
  if (!kProgress.value.total) return 0
  return Math.min(100, Math.round((kProgress.value.completed / kProgress.value.total) * 100))
})

const kProgressLabel = computed(() => {
  if (kProgress.value.status === 'completed') {
    return `评估完成：${kProgress.value.completed}/${kProgress.value.total}`
  }
  if (kProgress.value.status === 'failed') {
    return `评估失败：${kProgress.value.error || '未知错误'}`
  }
  const currentCategory = kProgress.value.current_category || '准备中'
  const currentK = kProgress.value.current_k === null ? '' : `，K=${kProgress.value.current_k}`
  return `正在读取 ${currentCategory}${currentK}：${kProgress.value.completed}/${kProgress.value.total}`
})

const kEvaluationDisplayTotal = computed(() => (
  kEvaluation.value.actual_sample_size
  || kEvaluation.value.paper_count
  || stats.value.paper_count
  || 0
))

const syncProgressPercent = computed(() => {
  if (syncProgress.value.status === 'completed') return 100
  if (syncProgress.value.status === 'failed') return 100
  if (syncProgress.value.stage === 'fetching') return 18
  if (syncProgress.value.stage === 'storing') return 42
  if (syncProgress.value.stage === 'embedding') {
    const totalCount = Number(syncProgress.value.embedding_total || 0)
    if (!totalCount) return 68
    const completed = Number(syncProgress.value.embedding_completed || 0)
    return Math.min(96, 50 + Math.round((completed / totalCount) * 44))
  }
  return 6
})

const syncProgressLabel = computed(() => {
  if (syncProgress.value.status === 'completed') {
    return `更新完成：获取 ${syncProgress.value.fetched} 篇，新增 ${syncProgress.value.inserted} 篇，生成向量 ${syncProgress.value.embedded} 篇`
  }
  if (syncProgress.value.status === 'failed') {
    return `更新失败：${syncProgress.value.error || syncProgress.value.message || '未知错误'}`
  }
  if (syncProgress.value.stage === 'fetching') {
    return `正在查询 arXiv，已获取 ${syncProgress.value.fetched || 0} 篇`
  }
  if (syncProgress.value.stage === 'storing') {
    return `正在写入数据库：获取 ${syncProgress.value.fetched || 0} 篇`
  }
  if (syncProgress.value.stage === 'embedding') {
    const totalCount = Number(syncProgress.value.embedding_total || 0)
    if (!totalCount) return '正在检查新增论文向量'
    return `正在生成 embedding：${syncProgress.value.embedding_completed || 0}/${totalCount}`
  }
  return '正在准备增量更新'
})

function categoryCount(id) {
  return Number(categories.value.find((item) => item.id === id)?.count || 0)
}

function showError(error) {
  const detail = error?.response?.data?.detail || error?.response?.data?.message || error.message
  statusText.value = `操作失败：${detail}`
}

function markClickedButton(event) {
  const button = event.target.closest('button')
  if (!button || button.disabled) return
  button.classList.add('clicked')
  if (clickedButtonTimer) clearTimeout(clickedButtonTimer)
  clickedButtonTimer = setTimeout(() => {
    button.classList.remove('clicked')
    clickedButtonTimer = null
  }, 1000)
}

function timeFilterParams() {
  return {
    time_field: timeField.value,
    date_from: dateFrom.value || undefined,
    date_to: dateTo.value || undefined
  }
}

function setKChartRef(id, element) {
  if (element) {
    kChartElements.set(id, element)
    nextTick(renderAllKCharts)
  } else {
    kChartElements.delete(id)
  }
}

function setClusterChartRef(id, element) {
  if (element) {
    clusterChartElements.set(id, element)
    nextTick(renderAllClusterCharts)
  } else {
    clusterChartElements.delete(id)
  }
}

function mergeSelectedDefaults(ids) {
  const merged = { ...selectedKs.value }
  for (const id of ids) {
    if (!merged[id]) merged[id] = defaultK
  }
  selectedKs.value = merged
}

function syncSelectedFromStats(payload) {
  if (!payload?.selected) return
  selectedKs.value = { ...selectedKs.value, ...payload.selected }
}

async function clearDateFilter() {
  dateFrom.value = ''
  dateTo.value = ''
  await searchPapers()
}

async function loadCategories() {
  try {
    const response = await axios.get('/api/categories')
    categories.value = response.data.items || []
    mergeSelectedDefaults(categories.value.map((item) => item.id))
  } catch (error) {
    showError(error)
  }
}

async function searchPapers() {
  loading.value = true
  try {
    const response = await axios.get('/api/papers', {
      params: {
        query: searchQuery.value || undefined,
        category: category.value || undefined,
        ...timeFilterParams(),
        page: 1,
        page_size: 30
      }
    })
    papers.value = response.data.items
    total.value = response.data.total
  } catch (error) {
    showError(error)
  } finally {
    loading.value = false
  }
}

async function selectPaper(id) {
  try {
    const response = await axios.get(`/api/papers/${id}`)
    selectedPaper.value = response.data
  } catch (error) {
    showError(error)
  }
}

async function recommendByPaper(id) {
  try {
    const response = await axios.get(`/api/papers/${id}/recommendations`, {
      params: { top_k: 8, ...timeFilterParams() }
    })
    recommendations.value = response.data.items
    statusText.value = '已根据选中论文生成推荐'
  } catch (error) {
    showError(error)
  }
}

async function recommendByQuery() {
  try {
    const response = await axios.post('/api/recommendations/query', {
      query: queryText.value,
      top_k: 8,
      time_field: timeField.value,
      date_from: dateFrom.value || null,
      date_to: dateTo.value || null
    })
    recommendations.value = response.data.items
    statusText.value = '已根据研究兴趣生成推荐'
  } catch (error) {
    showError(error)
  }
}

function clearKProgressTimer() {
  if (kProgressTimer) {
    clearInterval(kProgressTimer)
    kProgressTimer = null
  }
}

function clearSyncProgressTimer() {
  if (syncProgressTimer) {
    clearInterval(syncProgressTimer)
    syncProgressTimer = null
  }
}

function updateSyncProgress(jobState) {
  const result = jobState.result || {}
  syncProgress.value = {
    status: jobState.status || 'running',
    stage: jobState.stage || 'queued',
    fetched: Number(jobState.fetched ?? result.fetched ?? 0),
    inserted: Number(jobState.inserted ?? result.inserted ?? 0),
    skipped: Number(jobState.skipped ?? result.skipped ?? 0),
    embedded: Number(jobState.embedded ?? result.embedded ?? 0),
    embedding_completed: Number(jobState.embedding_completed ?? 0),
    embedding_total: Number(jobState.embedding_total ?? 0),
    message: result.message || jobState.message || '',
    error: jobState.error || null
  }
}

async function finishIncrementalUpdate(jobState) {
  clearSyncProgressTimer()
  updateSyncProgress(jobState)
  syncing.value = false
  statusText.value = jobState.result?.message || syncProgress.value.message || '数据更新完成'
  await loadAll()
  await loadCategories()
}

function pollIncrementalUpdate(jobId) {
  clearSyncProgressTimer()
  syncProgressTimer = setInterval(async () => {
    try {
      const response = await axios.get(`/api/sync/arxiv-incremental/status/${jobId}`)
      updateSyncProgress(response.data)
      if (response.data.status === 'completed') {
        await finishIncrementalUpdate(response.data)
      } else if (response.data.status === 'failed') {
        clearSyncProgressTimer()
        syncing.value = false
        statusText.value = `数据更新失败：${response.data.error || '未知错误'}`
      }
    } catch (error) {
      clearSyncProgressTimer()
      syncing.value = false
      showError(error)
    }
  }, 1500)
}

async function startIncrementalUpdate() {
  clearSyncProgressTimer()
  syncing.value = true
  syncProgress.value = {
    status: 'running',
    stage: 'queued',
    fetched: 0,
    inserted: 0,
    skipped: 0,
    embedded: 0,
    embedding_completed: 0,
    embedding_total: 0,
    message: '',
    error: null
  }
  statusText.value = '正在启动 arXiv 手动增量更新...'
  try {
    const response = await axios.post('/api/sync/arxiv-incremental/start')
    updateSyncProgress(response.data)
    if (response.data.status === 'completed') {
      await finishIncrementalUpdate(response.data)
    } else if (response.data.status === 'failed') {
      syncing.value = false
      statusText.value = `数据更新失败：${response.data.error || '未知错误'}`
    } else {
      pollIncrementalUpdate(response.data.job_id)
    }
  } catch (error) {
    syncing.value = false
    showError(error)
  }
}

function updateKProgress(jobState) {
  kProgress.value = {
    status: jobState.status || 'running',
    completed: Number(jobState.completed || 0),
    total: Number(jobState.total || defaultCategoryOrder.length * 12),
    current_k: jobState.current_k ?? null,
    current_category: jobState.current_category || null,
    error: jobState.error || null
  }
}

async function finishKEvaluation(jobState) {
  clearKProgressTimer()
  evaluatingK.value = false
  const result = jobState.result
  if (!result) {
    statusText.value = 'K 值评估完成，但后端没有返回折线图数据'
    return
  }
  kEvaluation.value = result
  mergeSelectedDefaults((result.categories || []).map((item) => item.id))
  await nextTick()
  renderAllKCharts()
  statusText.value = `已读取分组聚类缓存，包含 ${kEvaluation.value.actual_sample_size} 篇论文`
}

function pollKProgress(jobId) {
  clearKProgressTimer()
  kProgressTimer = setInterval(async () => {
    try {
      const response = await axios.get(`/api/clusters/evaluate-k/status/${jobId}`)
      updateKProgress(response.data)
      if (response.data.status === 'completed') {
        await finishKEvaluation(response.data)
      } else if (response.data.status === 'failed') {
        clearKProgressTimer()
        evaluatingK.value = false
        statusText.value = `K 值评估失败：${response.data.error || '未知错误'}`
      }
    } catch (error) {
      clearKProgressTimer()
      evaluatingK.value = false
      showError(error)
    }
  }, 1200)
}

async function confirmReevaluateK() {
  const confirmed = window.confirm('将重新读取 cluster_cache/manifest.json 中的分组 K 值评估结果。确定继续吗？')
  if (!confirmed) {
    statusText.value = '已取消刷新 K 值评估'
    return
  }
  await evaluateK(true)
}

async function evaluateK(force = false) {
  clearKProgressTimer()
  evaluatingK.value = true
  kProgress.value = {
    status: 'running',
    completed: 0,
    total: defaultCategoryOrder.length * 12,
    current_k: null,
    current_category: null,
    error: null
  }
  statusText.value = force
    ? '正在重新读取分组聚类缓存中的 K 值评估结果...'
    : '正在读取分组聚类缓存中的 K 值评估结果...'
  try {
    const response = await axios.post('/api/clusters/evaluate-k/start', { force })
    updateKProgress(response.data)
    if (response.data.status === 'completed') {
      await finishKEvaluation(response.data)
    } else if (response.data.status === 'failed') {
      evaluatingK.value = false
      statusText.value = `K 值评估失败：${response.data.error || '未知错误'}`
    } else {
      pollKProgress(response.data.job_id)
    }
  } catch (error) {
    evaluatingK.value = false
    showError(error)
  }
}

function renderKChart(group) {
  const element = kChartElements.get(group.id)
  if (!element) return
  let chart = kCharts.get(group.id)
  if (!chart) {
    chart = echarts.init(element)
    kCharts.set(group.id, chart)
  }
  chart.off('click')
  chart.on('click', (params) => {
    const k = params.data?.[0]
    if (typeof k === 'number') {
      selectedKs.value = { ...selectedKs.value, [group.id]: k }
      renderKChart(group)
      renderAllClusterCharts()
    }
  })

  const seriesData = (group.items || [])
    .filter((item) => item.silhouette_score !== null && item.silhouette_score !== undefined)
    .map((item) => [item.k, Number(item.silhouette_score)])
  const selectedPoint = seriesData.find((item) => item[0] === (selectedKs.value[group.id] || defaultK))

  chart.setOption({
    tooltip: {
      formatter: (params) => {
        const data = params.data || []
        return `K=${data[0]}<br/>轮廓系数：${Number(data[1]).toFixed(4)}`
      }
    },
    grid: { left: 72, right: 24, top: 66, bottom: 42, containLabel: true },
    xAxis: { type: 'value', name: 'K', min: 8, max: 30, interval: 2 },
    yAxis: { type: 'value', name: '轮廓系数', nameLocation: 'middle', nameGap: 48, scale: true },
    series: [
      {
        name: '轮廓系数',
        type: 'line',
        symbolSize: 8,
        data: seriesData
      },
      {
        name: '当前选择',
        type: 'scatter',
        symbolSize: 16,
        data: selectedPoint ? [selectedPoint] : [],
        itemStyle: { color: '#b35c00' }
      }
    ]
  })
}

function renderAllKCharts() {
  for (const group of kEvaluationGroups.value) {
    renderKChart(group)
  }
}

async function applySelectedK() {
  applyingK.value = true
  statusText.value = '正在切换 8 个主分类的本地 KMeans 缓存...'
  try {
    const response = await axios.post('/api/clusters/apply-k', { selected: selectedKs.value })
    syncSelectedFromStats(response.data)
    await loadStatsAndPlot()
    statusText.value = `已应用分组 KMeans 缓存，共 ${response.data.cluster_count} 个子聚类`
  } catch (error) {
    showError(error)
  } finally {
    applyingK.value = false
  }
}

function renderClusterChart(group) {
  const element = clusterChartElements.get(group.id)
  if (!element) return
  let chart = clusterCharts.get(group.id)
  if (!chart) {
    chart = echarts.init(element)
    clusterCharts.set(group.id, chart)
  }
  chart.off('click')
  chart.on('click', (params) => {
    const id = params.data?.[3]
    if (id) selectPaper(id)
  })

  const points = group.points || []
  const seriesData = points.map((point) => [
    point.x,
    point.y,
    point.subcluster_id ?? point.cluster_id,
    point.id,
    point.title,
    point.primary_category || group.id
  ])

  chart.setOption({
    tooltip: {
      formatter: (params) => {
        const data = params.data || []
        return `${data[4]}<br/>主分类：${data[5]}<br/>子聚类：${data[2]}`
      }
    },
    grid: { left: 74, right: 74, top: 30, bottom: 48, containLabel: true },
    xAxis: { type: 'value', name: 'Dim 1', nameGap: 22 },
    yAxis: { type: 'value', name: 'Dim 2', nameGap: 34 },
    visualMap: {
      dimension: 2,
      min: 0,
      max: Math.max(1, ...points.map((point) => Number(point.subcluster_id ?? point.cluster_id ?? 0))),
      right: 6,
      top: 18,
      calculable: true,
      show: points.length > 0
    },
    series: [
      {
        type: 'scatter',
        symbolSize: 8,
        data: seriesData
      }
    ]
  }, true)
}

function renderAllClusterCharts() {
  for (const group of clusterGroupsForDisplay.value) {
    renderClusterChart(group)
  }
}

async function loadStatsAndPlot() {
  const [statsResponse, plotResponse] = await Promise.all([
    axios.get('/api/stats'),
    axios.get('/api/clusters/plot')
  ])
  stats.value = statsResponse.data
  syncSelectedFromStats(statsResponse.data)
  clusterGroups.value = plotResponse.data.groups || []
  mergeSelectedDefaults(clusterGroups.value.map((item) => item.id))
  await nextTick()
  renderAllClusterCharts()
}

async function loadAll() {
  await searchPapers()
  try {
    await loadStatsAndPlot()
  } catch (error) {
    showError(error)
  }
}

function resizeCharts() {
  for (const chart of kCharts.values()) chart.resize()
  for (const chart of clusterCharts.values()) chart.resize()
}

onMounted(async () => {
  document.addEventListener('click', markClickedButton)
  window.addEventListener('resize', resizeCharts)
  await loadCategories()
  await loadAll()
})

onBeforeUnmount(() => {
  clearKProgressTimer()
  clearSyncProgressTimer()
  document.removeEventListener('click', markClickedButton)
  window.removeEventListener('resize', resizeCharts)
  if (clickedButtonTimer) clearTimeout(clickedButtonTimer)
  for (const chart of kCharts.values()) chart.dispose()
  for (const chart of clusterCharts.values()) chart.dispose()
})
</script>
