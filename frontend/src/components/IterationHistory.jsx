import React, { useState, useEffect } from 'react';
import './IterationHistory.css';

const IterationHistory = () => {
  const [iterations, setIterations] = useState([]);
  const [selectedIteration, setSelectedIteration] = useState(null);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchIterations();
  }, [filter]);

  const fetchIterations = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/iterations?filter=${filter}`);
      const data = await response.json();
      setIterations(data);
    } catch (error) {
      console.error('Failed to fetch iterations:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return '✅';
      case 'partial':
        return '⚠️';
      case 'failed':
        return '❌';
      case 'running':
        return '🔄';
      default:
        return '⏸️';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return '#4CAF50';
      case 'partial':
        return '#FFC107';
      case 'failed':
        return '#f44336';
      case 'running':
        return '#2196F3';
      default:
        return '#9E9E9E';
    }
  };

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const IterationDetail = ({ iteration }) => (
    <div className="iteration-detail">
      <h3>Iteration #{iteration.number} Details</h3>

      <div className="detail-section">
        <h4>Objectives</h4>
        <ul>
          {iteration.objectives?.map((obj, idx) => (
            <li key={idx} className={obj.completed ? 'completed' : 'pending'}>
              {obj.completed ? '✓' : '○'} {obj.description}
            </li>
          ))}
        </ul>
      </div>

      <div className="detail-section">
        <h4>Changes Made</h4>
        <div className="changes-grid">
          <div className="change-stat">
            <span className="stat-label">Files Modified:</span>
            <span className="stat-value">{iteration.filesModified || 0}</span>
          </div>
          <div className="change-stat">
            <span className="stat-label">Lines Added:</span>
            <span className="stat-value green">+{iteration.linesAdded || 0}</span>
          </div>
          <div className="change-stat">
            <span className="stat-label">Lines Removed:</span>
            <span className="stat-value red">-{iteration.linesRemoved || 0}</span>
          </div>
          <div className="change-stat">
            <span className="stat-label">Tests Added:</span>
            <span className="stat-value">{iteration.testsAdded || 0}</span>
          </div>
        </div>
      </div>

      <div className="detail-section">
        <h4>Metrics Improvement</h4>
        <div className="metrics-changes">
          {iteration.metricsChanges?.map((change, idx) => (
            <div key={idx} className="metric-change">
              <span className="metric-name">{change.name}:</span>
              <span className={`metric-delta ${change.improved ? 'positive' : 'negative'}`}>
                {change.improved ? '↑' : '↓'} {Math.abs(change.delta)}%
              </span>
              <span className="metric-final">{change.finalValue}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="detail-section">
        <h4>Agent Activity</h4>
        <div className="agent-activity">
          {iteration.agentActivity?.map((activity, idx) => (
            <div key={idx} className="activity-item">
              <span className="agent-name">{activity.agent}:</span>
              <span className="task-count">{activity.tasksCompleted} tasks</span>
              <span className="success-rate">({activity.successRate}% success)</span>
            </div>
          ))}
        </div>
      </div>

      {iteration.errors && iteration.errors.length > 0 && (
        <div className="detail-section">
          <h4>Errors Encountered</h4>
          <div className="errors-list">
            {iteration.errors.map((error, idx) => (
              <div key={idx} className="error-item">
                <span className="error-type">{error.type}:</span>
                <span className="error-message">{error.message}</span>
                <span className="error-resolution">{error.resolved ? '✓ Resolved' : '⚠️ Pending'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="detail-actions">
        <button className="btn-secondary" onClick={() => setSelectedIteration(null)}>
          Back to List
        </button>
        <button className="btn-primary">
          View Full Report
        </button>
        <button className="btn-primary">
          Replay Iteration
        </button>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="iteration-history loading">
        <div className="loader"></div>
        <p>Loading iteration history...</p>
      </div>
    );
  }

  return (
    <div className="iteration-history">
      {selectedIteration ? (
        <IterationDetail iteration={selectedIteration} />
      ) : (
        <>
          <div className="history-header">
            <h2>Iteration History</h2>
            <div className="history-controls">
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Iterations</option>
                <option value="success">Successful</option>
                <option value="failed">Failed</option>
                <option value="recent">Last 24 Hours</option>
                <option value="milestone">Milestones Only</option>
              </select>
              <button className="btn-refresh" onClick={fetchIterations}>
                🔄 Refresh
              </button>
            </div>
          </div>

          <div className="iterations-timeline">
            {iterations.map((iteration) => (
              <div
                key={iteration.id}
                className={`iteration-card ${iteration.status}`}
                onClick={() => setSelectedIteration(iteration)}
              >
                <div className="iteration-header">
                  <span className="iteration-number">#{iteration.number}</span>
                  <span
                    className="iteration-status"
                    style={{ color: getStatusColor(iteration.status) }}
                  >
                    {getStatusIcon(iteration.status)} {iteration.status}
                  </span>
                  <span className="iteration-time">{new Date(iteration.timestamp).toLocaleString()}</span>
                </div>

                <div className="iteration-body">
                  <div className="iteration-metrics">
                    <div className="metric">
                      <span className="label">Duration:</span>
                      <span className="value">{formatDuration(iteration.duration)}</span>
                    </div>
                    <div className="metric">
                      <span className="label">Tasks:</span>
                      <span className="value">{iteration.tasksCompleted}/{iteration.tasksTotal}</span>
                    </div>
                    <div className="metric">
                      <span className="label">Quality:</span>
                      <span className="value">{iteration.qualityScore}%</span>
                    </div>
                  </div>

                  {iteration.description && (
                    <div className="iteration-description">
                      {iteration.description}
                    </div>
                  )}

                  {iteration.milestone && (
                    <div className="iteration-milestone">
                      🎯 Milestone: {iteration.milestone}
                    </div>
                  )}
                </div>

                <div className="iteration-agents">
                  {iteration.activeAgents?.map(agent => (
                    <span key={agent} className="agent-badge">{agent}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {iterations.length === 0 && (
            <div className="no-iterations">
              <p>No iterations found for the selected filter.</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default IterationHistory;