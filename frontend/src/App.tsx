import { FileText, MessageSquareText } from 'lucide-react'
import { NavLink, Navigate, Route, Routes } from 'react-router'
import { ChatPage } from './pages/ChatPage'
import { ImportPage } from './pages/ImportPage'

export function App() {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="site-header">
        <NavLink className="brand" to="/import" aria-label="ContextForge home">
          <span className="brand-mark" aria-hidden="true"><FileText size={21} strokeWidth={2} /></span>
          <span>ContextForge</span>
        </NavLink>
        <nav className="site-nav" aria-label="Main navigation">
          <NavLink to="/import" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            <FileText size={17} aria-hidden="true" />Import documents
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
            <MessageSquareText size={17} aria-hidden="true" />Ask questions
          </NavLink>
        </nav>
      </header>
      <main id="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/import" replace />} />
          <Route path="/import" element={<ImportPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="*" element={<Navigate to="/import" replace />} />
        </Routes>
      </main>
    </div>
  )
}
