import React from 'react';
import './AgentStatus.css';

const AgentStatus = ({ agents, detailed = false }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'working':
        return '#4CAF50';
      case 'idle':
        return '#FFC107';
      case 'error':
        return '#f44336';
      default:
        return '#9E9E9E';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'working':
        return '⚡';
      case 'idle':
        return '💤';
      case 'error':
        return '❌';
      default:
        return '❓';
    }
  };

  if (!agents || agents.length === 0) {
    return (
      <div className="agent-status-container">
        <h2>Agent Status</h2>
        <div className="no-agents">No agents connected</div>
      </div>
    );
  }

  return (
    <div className="agent-status-container">
      <h2>Agent Status</h2>
      <div className={`agents-grid ${detailed ? 'detailed' : 'compact'}`}>
        {agents.map(agent => (
          <div key={agent.id} className="agent-card">
            <div className="agent-header">
              <span className="status-icon">{getStatusIcon(agent.status)}</span>
              <h3>{agent.name}</h3>
              <span
                className="status-badge"
                style={{ backgroundColor: getStatusColor(agent.status) }}
              >
                {agent.status}
              </span>
            </div>

            <div className="agent-body">
              <div className="agent-metric">
                <span className="metric-label">Current Task:</span>
                <span className="metric-value">{agent.currentTask || 'None'}</span>
              </div>

              <div className="agent-metric">
                <span className="metric-label">Tasks Completed:</span>
                <span className="metric-value">{agent.tasksCompleted}</span>
              </div>

              <div className="agent-metric">
                <span className="metric-label">Success Rate:</span>
                <span className="metric-value">{agent.successRate}%</span>
              </div>

              {detailed && (
                <>
                  <div className="agent-metric">
                    <span className="metric-label">Agent ID:</span>
                    <span className="metric-value mono">{agent.id}</span>
                  </div>

                  <div className="agent-capabilities">
                    <span className="metric-label">Capabilities:</span>
                    <div className="capabilities-list">
                      {agent.capabilities?.map(cap => (
                        <span key={cap} className="capability-tag">{cap}</span>
                      ))}
                    </div>
                  </div>

                  <div className="agent-history">
                    <span className="metric-label">Recent Actions:</span>
                    <div className="history-list">
                      {agent.recentActions?.slice(0, 5).map((action, idx) => (
                        <div key={idx} className="history-item">
                          <span className="history-time">{action.time}</span>
                          <span className="history-action">{action.description}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

            {detailed && (
              <div className="agent-footer">
                <button className="btn-small">View Logs</button>
                <button className="btn-small">Restart</button>
                <button className="btn-small danger">Stop</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentStatus;