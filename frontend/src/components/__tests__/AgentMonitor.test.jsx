import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import AgentMonitor from '../AgentMonitor';
import { ProjectProvider } from '../../contexts/ProjectContext';

global.fetch = vi.fn();

describe('AgentMonitor Component', () => {
  const mockAgents = [
    {
      id: 'agent-1',
      name: 'Coder Agent',
      status: 'active',
      current_task: 'Implementing feature X',
      tasks_completed: 15,
      success_rate: 0.93
    },
    {
      id: 'agent-2',
      name: 'Tester Agent',
      status: 'idle',
      current_task: null,
      tasks_completed: 8,
      success_rate: 1.0
    }
  ];

  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders agent monitor title', () => {
    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    expect(screen.getByText(/agent monitor/i)).toBeInTheDocument();
  });

  test('displays agent status information', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAgents
    });

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Coder Agent')).toBeInTheDocument();
      expect(screen.getByText('Tester Agent')).toBeInTheDocument();
    });
  });

  test('shows active agent status correctly', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAgents
    });

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/active/i)).toBeInTheDocument();
      expect(screen.getByText('Implementing feature X')).toBeInTheDocument();
    });
  });

  test('displays agent metrics', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAgents
    });

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/15/)).toBeInTheDocument(); // tasks completed
      expect(screen.getByText(/93%/)).toBeInTheDocument(); // success rate
    });
  });

  test('handles real-time updates via WebSocket', async () => {
    const mockWebSocket = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      close: vi.fn()
    };

    global.WebSocket = vi.fn(() => mockWebSocket);

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAgents
    });

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(mockWebSocket.addEventListener).toHaveBeenCalledWith(
        'message',
        expect.any(Function)
      );
    });
  });

  test('shows idle agents differently', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockAgents
    });

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      const idleElements = screen.getAllByText(/idle/i);
      expect(idleElements.length).toBeGreaterThan(0);
    });
  });

  test('handles error state gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('Connection failed'));

    render(
      <ProjectProvider>
        <AgentMonitor />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/error|failed/i)).toBeInTheDocument();
    });
  });
});
