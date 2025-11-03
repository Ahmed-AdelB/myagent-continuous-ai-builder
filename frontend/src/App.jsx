import React, { useState, useEffect } from 'react'
import './App.css'
import Dashboard from './components/Dashboard'
import ProjectManager from './components/ProjectManager'
import AgentMonitor from './components/AgentMonitor'
import MetricsView from './components/MetricsView'
import { WebSocketProvider } from './contexts/WebSocketContext'
import { ProjectProvider } from './contexts/ProjectContext'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    // Check API connection
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(() => setConnected(true))
      .catch(() => setConnected(false))
  }, [])

  return (
    <WebSocketProvider>
      <ProjectProvider>
        <div className="app">
          <header className="app-header">
            <h1>🤖 Continuous AI Builder</h1>
            <div className="connection-status">
              <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}></span>
              {connected ? 'Connected' : 'Disconnected'}
            </div>
          </header>

          <nav className="app-nav">
            <button 
              className={activeTab === 'dashboard' ? 'active' : ''}
              onClick={() => setActiveTab('dashboard')}
            >
              Dashboard
            </button>
            <button 
              className={activeTab === 'projects' ? 'active' : ''}
              onClick={() => setActiveTab('projects')}
            >
              Projects
            </button>
            <button 
              className={activeTab === 'agents' ? 'active' : ''}
              onClick={() => setActiveTab('agents')}
            >
              Agents
            </button>
            <button 
              className={activeTab === 'metrics' ? 'active' : ''}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics
            </button>
          </nav>

          <main className="app-content">
            {activeTab === 'dashboard' && <Dashboard />}
            {activeTab === 'projects' && <ProjectManager />}
            {activeTab === 'agents' && <AgentMonitor />}
            {activeTab === 'metrics' && <MetricsView />}
          </main>

          <footer className="app-footer">
            <p>Never stops until perfection is achieved</p>
          </footer>
        </div>
      </ProjectProvider>
    </WebSocketProvider>
  )
}

export default App