import React, { useState, useEffect } from 'react'
import AgentStatus from './AgentStatus'

function AgentMonitor() {
  const [agents, setAgents] = useState([])
  const [selectedAgent, setSelectedAgent] = useState(null)

  useEffect(() => {
    fetchAgents()
    const interval = setInterval(fetchAgents, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchAgents = async () => {
    try {
      // Get first project (for simplicity)
      const projectsResponse = await fetch('http://localhost:8000/projects')
      const projects = await projectsResponse.json()

      if (projects.length > 0) {
        const projectId = projects[0].id
        const response = await fetch(`http://localhost:8000/projects/${projectId}/agents`)
        const data = await response.json()
        setAgents(data)
      }
    } catch (error) {
      console.error('Failed to fetch agents:', error)
    }
  }

  return (
    <div className="agent-monitor">
      <h2>Agent Monitor</h2>
      <p className="subtitle">Real-time monitoring of all AI agents</p>

      <div className="agents-grid">
        {agents.map((agent, index) => (
          <div
            key={agent.name}
            className={`agent-card ${selectedAgent === agent.name ? 'selected' : ''}`}
            onClick={() => setSelectedAgent(agent.name)}
          >
            <div className="agent-header">
              <h3>{agent.name}</h3>
              <span className={`status-badge ${agent.status}`}>{agent.status}</span>
            </div>
            <p className="agent-role">{agent.role}</p>
            {agent.capabilities && agent.capabilities.length > 0 && (
              <div className="capabilities">
                <strong>Capabilities:</strong>
                <ul>
                  {agent.capabilities.map((cap, i) => (
                    <li key={i}>{cap}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      {agents.length === 0 && (
        <div className="empty-state">
          <p>No active agents found. Create a project to initialize agents.</p>
        </div>
      )}

      <style jsx>{`
        .agent-monitor {
          padding: 20px;
        }
        .subtitle {
          color: #888;
          margin-bottom: 20px;
        }
        .agents-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 20px;
        }
        .agent-card {
          background: #1e1e1e;
          padding: 20px;
          border-radius: 8px;
          border: 2px solid #333;
          cursor: pointer;
          transition: all 0.2s;
        }
        .agent-card:hover {
          border-color: #007bff;
        }
        .agent-card.selected {
          border-color: #007bff;
          background: #252525;
        }
        .agent-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        .status-badge {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          text-transform: uppercase;
        }
        .status-badge.active {
          background: #28a745;
          color: white;
        }
        .status-badge.idle {
          background: #6c757d;
          color: white;
        }
        .status-badge.working {
          background: #007bff;
          color: white;
        }
        .agent-role {
          color: #888;
          margin: 10px 0;
        }
        .capabilities {
          margin-top: 15px;
          font-size: 14px;
        }
        .capabilities ul {
          margin: 5px 0 0 20px;
          color: #aaa;
        }
        .empty-state {
          text-align: center;
          padding: 60px 20px;
          color: #666;
        }
      `}</style>
    </div>
  )
}

export default AgentMonitor
