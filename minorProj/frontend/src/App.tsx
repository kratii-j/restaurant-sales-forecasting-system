import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Forecasting from './pages/Forecasting'
import RiskAnalysis from './pages/RiskAnalysis'
import RestaurantExplorer from './pages/RestaurantExplorer'
import ModelPerformance from './pages/ModelPerformance'

type Page = 'dashboard' | 'forecasting' | 'risk' | 'restaurants' | 'models'

const VALID: Page[] = ['dashboard', 'forecasting', 'risk', 'restaurants', 'models']

function getPage(): Page {
  const h = window.location.hash.slice(1)
  return VALID.includes(h as Page) ? (h as Page) : 'dashboard'
}

function App() {
  const [page, setPage] = useState<Page>(getPage)

  useEffect(() => {
    const fn = () => setPage(getPage())
    window.addEventListener('hashchange', fn)
    return () => window.removeEventListener('hashchange', fn)
  }, [])

  const nav = (p: Page) => {
    window.location.hash = p
    setPage(p)
  }

  const render = () => {
    switch (page) {
      case 'dashboard': return <Dashboard />
      case 'forecasting': return <Forecasting />
      case 'risk': return <RiskAnalysis />
      case 'restaurants': return <RestaurantExplorer />
      case 'models': return <ModelPerformance />
    }
  }

  return (
    <>
      <Sidebar active={page} onNavigate={nav} />
      <main className="main-content">{render()}</main>
    </>
  )
}

export default App
