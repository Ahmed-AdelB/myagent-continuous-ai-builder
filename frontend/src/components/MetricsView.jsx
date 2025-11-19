import React, { useState, useEffect } from 'react'
import MetricsPanel from './MetricsPanel'
import ErrorAnalytics from './ErrorAnalytics'
import IterationHistory from './IterationHistory'

function MetricsView() {
  const [activeView, setActiveView] = useState('overview')

  return (
    <div className="metrics-view">
      <div className="metrics-header">
        <h2>Metrics & Analytics</h2>
        <div className="view-selector">
          <button
            className={activeView === 'overview' ? 'active' : ''}
            onClick={() => setActiveView('overview')}
          >
            Overview
          </button>
          <button
            className={activeView === 'errors' ? 'active' : ''}
            onClick={() => setActiveView('errors')}
          >
            Error Analysis
          </button>
          <button
            className={activeView === 'iterations' ? 'active' : ''}
            onClick={() => setActiveView('iterations')}
          >
            Iterations
          </button>
        </div>
      </div>

      <div className="metrics-content">
        {activeView === 'overview' && (
          <div>
            <MetricsPanel />
          </div>
        )}

        {activeView === 'errors' && (
          <div>
            <ErrorAnalytics />
          </div>
        )}

        {activeView === 'iterations' && (
          <div>
            <IterationHistory />
          </div>
        )}
      </div>

      <style jsx>{`
        .metrics-view {
          padding: 20px;
        }
        .metrics-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        .view-selector {
          display: flex;
          gap: 10px;
        }
        .view-selector button {
          padding: 8px 16px;
          background: #2d2d2d;
          border: 1px solid #444;
          border-radius: 4px;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }
        .view-selector button:hover {
          background: #3d3d3d;
        }
        .view-selector button.active {
          background: #007bff;
          border-color: #007bff;
        }
        .metrics-content {
          min-height: 500px;
        }
      `}</style>
    </div>
  )
}

export default MetricsView
