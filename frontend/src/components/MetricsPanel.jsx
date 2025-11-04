import React, { useState, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './MetricsPanel.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const MetricsPanel = () => {
  const [selectedMetric, setSelectedMetric] = useState('performance');
  const [timeRange, setTimeRange] = useState('24h');
  const [metricsData, setMetricsData] = useState({
    performance: [],
    testCoverage: [],
    bugCount: [],
    codeQuality: []
  });

  useEffect(() => {
    // Fetch metrics data
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/metrics?range=${timeRange}`);
        const data = await response.json();
        setMetricsData(data);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [timeRange]);

  const getChartData = () => {
    const data = metricsData[selectedMetric] || [];

    return {
      labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
      datasets: [
        {
          label: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
          data: data.map(d => d.value),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4
        }
      ]
    };
  };

  const getQualityDistribution = () => {
    return {
      labels: ['Excellent', 'Good', 'Fair', 'Poor'],
      datasets: [
        {
          data: [45, 30, 15, 10],
          backgroundColor: [
            'rgba(75, 192, 192, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(255, 99, 132, 0.8)'
          ]
        }
      ]
    };
  };

  const getAgentPerformance = () => {
    return {
      labels: ['Coder', 'Tester', 'Debugger', 'Architect', 'Analyzer', 'UI Refiner'],
      datasets: [
        {
          label: 'Tasks Completed',
          data: [65, 59, 80, 81, 56, 55],
          backgroundColor: 'rgba(75, 192, 192, 0.6)'
        },
        {
          label: 'Success Rate',
          data: [95, 92, 88, 94, 97, 90],
          backgroundColor: 'rgba(54, 162, 235, 0.6)'
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#333'
        }
      },
      title: {
        display: false
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          color: '#666'
        }
      },
      y: {
        grid: {
          color: 'rgba(0,0,0,0.1)'
        },
        ticks: {
          color: '#666'
        }
      }
    }
  };

  const metricCards = [
    {
      title: 'Avg Response Time',
      value: '234ms',
      change: '-12%',
      trend: 'positive'
    },
    {
      title: 'Error Rate',
      value: '0.3%',
      change: '-45%',
      trend: 'positive'
    },
    {
      title: 'Code Complexity',
      value: '7.2',
      change: '+2%',
      trend: 'negative'
    },
    {
      title: 'Test Pass Rate',
      value: '94.5%',
      change: '+3%',
      trend: 'positive'
    }
  ];

  return (
    <div className="metrics-panel">
      <div className="metrics-header">
        <h2>Performance Metrics</h2>
        <div className="metrics-controls">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="metric-selector"
          >
            <option value="performance">Performance</option>
            <option value="testCoverage">Test Coverage</option>
            <option value="bugCount">Bug Count</option>
            <option value="codeQuality">Code Quality</option>
          </select>

          <div className="time-range-selector">
            {['1h', '24h', '7d', '30d'].map(range => (
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

      <div className="metrics-cards">
        {metricCards.map((card, idx) => (
          <div key={idx} className="metric-card">
            <h4>{card.title}</h4>
            <div className="metric-value">{card.value}</div>
            <div className={`metric-change ${card.trend}`}>
              {card.trend === 'positive' ? '📈' : '📉'} {card.change}
            </div>
          </div>
        ))}
      </div>

      <div className="charts-grid">
        <div className="chart-container main-chart">
          <h3>Trend Analysis</h3>
          <Line data={getChartData()} options={chartOptions} />
        </div>

        <div className="chart-container">
          <h3>Quality Distribution</h3>
          <Doughnut
            data={getQualityDistribution()}
            options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                legend: {
                  position: 'bottom'
                }
              }
            }}
          />
        </div>

        <div className="chart-container">
          <h3>Agent Performance</h3>
          <Bar data={getAgentPerformance()} options={chartOptions} />
        </div>
      </div>

      <div className="metrics-insights">
        <h3>Key Insights</h3>
        <ul className="insights-list">
          <li className="insight positive">
            <span className="insight-icon">✅</span>
            Test coverage improved by 5% in the last 24 hours
          </li>
          <li className="insight positive">
            <span className="insight-icon">✅</span>
            Performance optimization reduced response time by 12%
          </li>
          <li className="insight warning">
            <span className="insight-icon">⚠️</span>
            Code complexity increased in 3 modules - review recommended
          </li>
          <li className="insight info">
            <span className="insight-icon">ℹ️</span>
            Next scheduled optimization in 2 hours
          </li>
        </ul>
      </div>
    </div>
  );
};

export default MetricsPanel;