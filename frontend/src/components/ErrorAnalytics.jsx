import React, { useState, useEffect } from 'react';
import { Line, Bar, Scatter, Pie } from 'react-chartjs-2';
import './ErrorAnalytics.css';

const ErrorAnalytics = () => {
  const [errors, setErrors] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [selectedError, setSelectedError] = useState(null);
  const [timeRange, setTimeRange] = useState('7d');
  const [errorStats, setErrorStats] = useState({
    total: 0,
    resolved: 0,
    pending: 0,
    critical: 0,
    recurring: 0
  });

  useEffect(() => {
    fetchErrorData();
  }, [timeRange]);

  const fetchErrorData = async () => {
    try {
      const [errorsRes, patternsRes, statsRes] = await Promise.all([
        fetch(`http://localhost:8000/api/errors?range=${timeRange}`),
        fetch(`http://localhost:8000/api/error-patterns`),
        fetch(`http://localhost:8000/api/error-stats`)
      ]);

      const errorsData = await errorsRes.json();
      const patternsData = await patternsRes.json();
      const statsData = await statsRes.json();

      setErrors(errorsData);
      setPatterns(patternsData);
      setErrorStats(statsData);
    } catch (error) {
      console.error('Failed to fetch error data:', error);
    }
  };

  const getErrorTrendData = () => {
    const grouped = errors.reduce((acc, error) => {
      const date = new Date(error.timestamp).toLocaleDateString();
      acc[date] = (acc[date] || 0) + 1;
      return acc;
    }, {});

    return {
      labels: Object.keys(grouped),
      datasets: [
        {
          label: 'Errors per Day',
          data: Object.values(grouped),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.4
        }
      ]
    };
  };

  const getErrorTypeDistribution = () => {
    const types = errors.reduce((acc, error) => {
      acc[error.type] = (acc[error.type] || 0) + 1;
      return acc;
    }, {});

    return {
      labels: Object.keys(types),
      datasets: [
        {
          data: Object.values(types),
          backgroundColor: [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)'
          ]
        }
      ]
    };
  };

  const getErrorByAgent = () => {
    const byAgent = errors.reduce((acc, error) => {
      acc[error.agent] = (acc[error.agent] || 0) + 1;
      return acc;
    }, {});

    return {
      labels: Object.keys(byAgent),
      datasets: [
        {
          label: 'Errors by Agent',
          data: Object.values(byAgent),
          backgroundColor: 'rgba(54, 162, 235, 0.6)'
        }
      ]
    };
  };

  const getErrorRecoveryTime = () => {
    const recoveryTimes = errors
      .filter(e => e.resolvedAt)
      .map(e => ({
        x: new Date(e.timestamp).getTime(),
        y: (new Date(e.resolvedAt).getTime() - new Date(e.timestamp).getTime()) / 1000 / 60
      }));

    return {
      datasets: [
        {
          label: 'Recovery Time (minutes)',
          data: recoveryTimes,
          backgroundColor: 'rgba(75, 192, 192, 0.6)'
        }
      ]
    };
  };

  const ErrorDetail = ({ error }) => (
    <div className="error-detail">
      <div className="error-detail-header">
        <h3>{error.type}: {error.message}</h3>
        <button
          className="close-btn"
          onClick={() => setSelectedError(null)}
        >
          ×
        </button>
      </div>

      <div className="error-detail-body">
        <div className="detail-grid">
          <div className="detail-item">
            <span className="label">Error ID:</span>
            <span className="value mono">{error.id}</span>
          </div>
          <div className="detail-item">
            <span className="label">Agent:</span>
            <span className="value">{error.agent}</span>
          </div>
          <div className="detail-item">
            <span className="label">Timestamp:</span>
            <span className="value">{new Date(error.timestamp).toLocaleString()}</span>
          </div>
          <div className="detail-item">
            <span className="label">Status:</span>
            <span className={`value status-${error.status}`}>{error.status}</span>
          </div>
          <div className="detail-item">
            <span className="label">Severity:</span>
            <span className={`value severity-${error.severity}`}>{error.severity}</span>
          </div>
          <div className="detail-item">
            <span className="label">Occurrences:</span>
            <span className="value">{error.count || 1}</span>
          </div>
        </div>

        <div className="stack-trace">
          <h4>Stack Trace</h4>
          <pre>{error.stackTrace}</pre>
        </div>

        {error.context && (
          <div className="error-context">
            <h4>Context</h4>
            <pre>{JSON.stringify(error.context, null, 2)}</pre>
          </div>
        )}

        {error.solution && (
          <div className="suggested-solution">
            <h4>Suggested Solution</h4>
            <p>{error.solution}</p>
            {error.similarErrors && (
              <div className="similar-errors">
                <h5>Similar errors resolved previously:</h5>
                <ul>
                  {error.similarErrors.map((similar, idx) => (
                    <li key={idx}>
                      {similar.message} - Resolved with: {similar.solution}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        <div className="error-actions">
          <button className="btn-primary">Mark as Resolved</button>
          <button className="btn-secondary">Assign to Agent</button>
          <button className="btn-secondary">Create Issue</button>
          <button className="btn-danger">Ignore</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="error-analytics">
      <div className="analytics-header">
        <h2>Error Analytics</h2>
        <div className="analytics-controls">
          <div className="time-range-selector">
            {['24h', '7d', '30d', 'all'].map(range => (
              <button
                key={range}
                className={`range-btn ${timeRange === range ? 'active' : ''}`}
                onClick={() => setTimeRange(range)}
              >
                {range}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="error-stats">
        <div className="stat-card">
          <span className="stat-label">Total Errors</span>
          <span className="stat-value">{errorStats.total}</span>
        </div>
        <div className="stat-card success">
          <span className="stat-label">Resolved</span>
          <span className="stat-value">{errorStats.resolved}</span>
          <span className="stat-percent">
            {errorStats.total > 0 ? Math.round(errorStats.resolved / errorStats.total * 100) : 0}%
          </span>
        </div>
        <div className="stat-card warning">
          <span className="stat-label">Pending</span>
          <span className="stat-value">{errorStats.pending}</span>
        </div>
        <div className="stat-card danger">
          <span className="stat-label">Critical</span>
          <span className="stat-value">{errorStats.critical}</span>
        </div>
        <div className="stat-card info">
          <span className="stat-label">Recurring</span>
          <span className="stat-value">{errorStats.recurring}</span>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3>Error Trend</h3>
          <Line
            data={getErrorTrendData()}
            options={{
              responsive: true,
              maintainAspectRatio: false
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Error Types</h3>
          <Pie
            data={getErrorTypeDistribution()}
            options={{
              responsive: true,
              maintainAspectRatio: false
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Errors by Agent</h3>
          <Bar
            data={getErrorByAgent()}
            options={{
              responsive: true,
              maintainAspectRatio: false
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Recovery Time</h3>
          <Scatter
            data={getErrorRecoveryTime()}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: {
                  type: 'linear',
                  position: 'bottom',
                  ticks: {
                    callback: function(value) {
                      return new Date(value).toLocaleDateString();
                    }
                  }
                }
              }
            }}
          />
        </div>
      </div>

      <div className="error-patterns">
        <h3>Detected Patterns</h3>
        <div className="patterns-grid">
          {patterns.map((pattern, idx) => (
            <div key={idx} className="pattern-card">
              <h4>{pattern.name}</h4>
              <p>{pattern.description}</p>
              <div className="pattern-stats">
                <span>Occurrences: {pattern.count}</span>
                <span>Last seen: {new Date(pattern.lastSeen).toLocaleDateString()}</span>
              </div>
              <div className="pattern-solution">
                <strong>Common Solution:</strong>
                <p>{pattern.solution}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="recent-errors">
        <h3>Recent Errors</h3>
        <div className="errors-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Message</th>
                <th>Agent</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {errors.slice(0, 10).map((error) => (
                <tr key={error.id}>
                  <td>{new Date(error.timestamp).toLocaleTimeString()}</td>
                  <td className={`error-type ${error.type}`}>{error.type}</td>
                  <td className="error-message">{error.message}</td>
                  <td>{error.agent}</td>
                  <td className={`status ${error.status}`}>{error.status}</td>
                  <td>
                    <button
                      className="btn-small"
                      onClick={() => setSelectedError(error)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selectedError && <ErrorDetail error={selectedError} />}
    </div>
  );
};

export default ErrorAnalytics;