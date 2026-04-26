const BASE = ''

async function fetchJson<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, opts)
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}

export const api = {
  dashboard: {
    summary: () => fetchJson<any>('/api/dashboard/summary'),
    trends: (days = 90) => fetchJson<any[]>(`/api/dashboard/trends?days=${days}`),
  },
  restaurants: {
    list: (params: Record<string, any> = {}) => {
      const qs = new URLSearchParams()
      Object.entries(params).forEach(([k, v]) => {
        if (v != null && v !== '') qs.set(k, String(v))
      })
      return fetchJson<any>(`/api/restaurants?${qs}`)
    },
    get: (id: number) => fetchJson<any>(`/api/restaurants/${id}`),
    forecast: (id: number) => fetchJson<any[]>(`/api/restaurants/${id}/forecast`),
  },
  predict: (data: any) =>
    fetchJson<any>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),
  risk: {
    summary: () => fetchJson<any>('/api/risk/summary'),
    restaurants: (params: Record<string, any> = {}) => {
      const qs = new URLSearchParams()
      Object.entries(params).forEach(([k, v]) => {
        if (v != null && v !== '') qs.set(k, String(v))
      })
      return fetchJson<any>(`/api/risk/restaurants?${qs}`)
    },
  },
  models: {
    comparison: () => fetchJson<any>('/api/models/comparison'),
    restaurantPerformance: (params: Record<string, any> = {}) => {
      const qs = new URLSearchParams()
      Object.entries(params).forEach(([k, v]) => {
        if (v != null && v !== '') qs.set(k, String(v))
      })
      return fetchJson<any>(`/api/models/restaurant-performance?${qs}`)
    },
  },
}
