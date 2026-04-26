interface StatusBadgeProps {
  level: string
}

export default function StatusBadge({ level }: StatusBadgeProps) {
  const n = level.toLowerCase()
  const cls = n === 'low' ? 'badge-low' : n === 'medium' ? 'badge-medium' : 'badge-high'
  return <span className={`badge ${cls}`}>{level}</span>
}
