import { useState } from 'react'

interface Column {
  key: string
  label: string
  render?: (value: any, row: any) => React.ReactNode
  sortable?: boolean
  align?: 'left' | 'center' | 'right'
}

interface DataTableProps {
  columns: Column[]
  data: any[]
  onRowClick?: (row: any) => void
}

export default function DataTable({ columns, data, onRowClick }: DataTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('asc') }
  }

  const sorted = [...data]
  if (sortKey) {
    sorted.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      if (av == null) return 1
      if (bv == null) return -1
      const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
      return sortDir === 'asc' ? cmp : -cmp
    })
  }

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map(col => (
              <th
                key={col.key}
                onClick={() => col.sortable !== false && handleSort(col.key)}
                style={{ cursor: col.sortable !== false ? 'pointer' : 'default', textAlign: col.align || 'left' }}
              >
                {col.label}
                {sortKey === col.key && <span className="sort-indicator">{sortDir === 'asc' ? ' ↑' : ' ↓'}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr key={i} onClick={() => onRowClick?.(row)} style={{ cursor: onRowClick ? 'pointer' : 'default' }}>
              {columns.map(col => (
                <td key={col.key} style={{ textAlign: col.align || 'left' }}>
                  {col.render ? col.render(row[col.key], row) : row[col.key] ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
