import React, { useState, useEffect } from 'react'
import { useProject } from '../contexts/ProjectContext'
import { useWebSocket } from '../contexts/WebSocketContext'
import './Dashboard.css'

function Dashboard() {
  const { currentProject, metrics } = useProject()
  const { isConnected, lastMessage } = useWebSocket()
  const [stats, setStats] = useState({
    activeProjects: 0,
    totalIterations: 0,
    averageQuality: 0,
    estimatedCompletion: null
  })

  useEffect(() => {
    // Update stats when project or metrics change
    if (currentProject) {
      setStats(prev => ({
        ...prev,
        totalIterations: currentProject.iteration || 0,
        averageQuality: metrics?.quality_score || 0
      }))
    }
  }, [currentProject, metrics])

  useEffect(() => {
    // Handle real-time updates
    if (lastMessage?.type === 'update') {
      setStats(prev => ({
        ...prev,
        ...lastMessage.data
      }))
    }
  }, [lastMessage])

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>System Dashboard</h2>
        <span className="live-indicator">
          {isConnected && '🔴 LIVE'}
        </span>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>Active Projects</h3>
          <div className="stat-value">{stats.activeProjects}</div>
        </div>

        <div className="stat-card">
          <h3>Total Iterations</h3>
          <div className="stat-value">{stats.totalIterations}</div>
        </div>

        <div className="stat-card">
          <h3>Quality Score</h3>
          <div className="stat-value">{stats.averageQuality.toFixed(1)}%</div>
          <div className="stat-bar">
            <div 
              className="stat-bar-fill"
              style={{width: `${stats.averageQuality}%`}}
            ></div>
          </div>
        </div>

        <div className="stat-card">
          <h3>Est. Completion</h3>
          <div className="stat-value">
            {stats.estimatedCompletion || 'Calculating...'}
          </div>
        </div>
      </div>

      {currentProject && (
        <div className="current-project">
          <h3>Current Project: {currentProject.name}</h3>
          <div className="project-details">
            <div className="detail-item">
              <span>State:</span>
              <span className={`state-badge state-${currentProject.state}`}>
                {currentProject.state}
              </span>
            </div>
            <div className="detail-item">
              <span>Iteration:</span>
              <span>{currentProject.iteration}</span>
            </div>
            <div className="detail-item">
              <span>Milestones:</span>
              <span>
                {currentProject.milestones?.completed || 0} / 
                {currentProject.milestones?.total || 0}
              </span>
            </div>
          </div>

          {metrics && (
            <div className="metrics-summary">
              <h4>Quality Metrics</h4>
              <div className="metrics-grid">
                <div className="metric">
                  <span>Test Coverage</span>
                  <span>{metrics.test_coverage?.toFixed(1) || 0}%</span>
                </div>
                <div className="metric">
                  <span>Critical Bugs</span>
                  <span>{metrics.critical_bugs || 0}</span>
                </div>
                <div className="metric">
                  <span>Performance</span>
                  <span>{metrics.performance_score?.toFixed(1) || 0}%</span>
                </div>
                <div className="metric">
                  <span>Documentation</span>
                  <span>{metrics.documentation_coverage?.toFixed(1) || 0}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="activity-feed">
        <h3>Recent Activity</h3>
        <div className="activity-list">
          {/* Activity items would be populated from WebSocket messages */}
          <div className="activity-item">
            <span className="activity-time">Just now</span>
            <span className="activity-text">System initialized</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard