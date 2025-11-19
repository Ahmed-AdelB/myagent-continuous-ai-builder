import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import MetricsPanel from '../MetricsPanel';
import { ProjectProvider } from '../../contexts/ProjectContext';

// Mock Chart.js
vi.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="line-chart">Line Chart</div>,
  Bar: () => <div data-testid="bar-chart">Bar Chart</div>,
  Doughnut: () => <div data-testid="doughnut-chart">Doughnut Chart</div>
}));

global.fetch = vi.fn();

describe('MetricsPanel Component', () => {
  const mockMetrics = {
    performance: [
      { timestamp: '2025-11-19T10:00:00Z', value: 85 },
      { timestamp: '2025-11-19T11:00:00Z', value: 90 }
    ],
    testCoverage: [
      { timestamp: '2025-11-19T10:00:00Z', value: 75 },
      { timestamp: '2025-11-19T11:00:00Z', value: 80 }
    ]
  };

  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders metrics panel with title', () => {
    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    expect(screen.getByText(/performance metrics/i)).toBeInTheDocument();
  });

  test('displays metric selector', () => {
    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    const selector = screen.getByRole('combobox', { name: /metric/i });
    expect(selector).toBeInTheDocument();
  });

  test('fetches and displays metrics data', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/metrics'),
        expect.any(Object)
      );
    });
  });

  test('renders charts correctly', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('doughnut-chart')).toBeInTheDocument();
    });
  });

  test('changes time range on button click', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    const timeRangeButton = screen.getByRole('button', { name: /7d/i });
    fireEvent.click(timeRangeButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('range=7d'),
        expect.any(Object)
      );
    });
  });

  test('switches between different metrics', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    const selector = screen.getByRole('combobox');
    fireEvent.change(selector, { target: { value: 'testCoverage' } });

    await waitFor(() => {
      expect(selector.value).toBe('testCoverage');
    });
  });

  test('displays metric cards with values', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/avg response time/i)).toBeInTheDocument();
      expect(screen.getByText(/error rate/i)).toBeInTheDocument();
    });
  });

  test('updates metrics periodically', async () => {
    vi.useFakeTimers();

    fetch.mockResolvedValue({
      ok: true,
      json: async () => mockMetrics
    });

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    // Advance 30 seconds
    vi.advanceTimersByTime(30000);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    vi.useRealTimers();
  });

  test('handles API error gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('API Error'));

    render(
      <ProjectProvider>
        <MetricsPanel />
      </ProjectProvider>
    );

    // Should not crash, console error logged
    await waitFor(() => {
      expect(screen.queryByText(/fatal error/i)).not.toBeInTheDocument();
    });
  });
});
