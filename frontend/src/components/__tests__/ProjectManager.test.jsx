import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import ProjectManager from '../ProjectManager';
import { ProjectProvider } from '../../contexts/ProjectContext';

// Mock fetch
global.fetch = vi.fn();

describe('ProjectManager Component', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders project manager with title', () => {
    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    expect(screen.getByText(/project manager/i)).toBeInTheDocument();
  });

  test('displays loading state initially', () => {
    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('fetches and displays projects', async () => {
    const mockProjects = [
      { id: '1', name: 'Test Project 1', status: 'running', progress: 50 },
      { id: '2', name: 'Test Project 2', status: 'completed', progress: 100 }
    ];

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockProjects
    });

    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Project 1')).toBeInTheDocument();
      expect(screen.getByText('Test Project 2')).toBeInTheDocument();
    });
  });

  test('handles create project action', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: '3', name: 'New Project', status: 'running' })
    });

    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    const createButton = screen.getByRole('button', { name: /create project/i });
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/projects'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  test('handles delete project action', async () => {
    const mockProjects = [
      { id: '1', name: 'Test Project', status: 'running', progress: 50 }
    ];

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockProjects
    });

    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Test Project')).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole('button', { name: /delete/i });

    fetch.mockResolvedValueOnce({ ok: true });

    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/projects/1'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  test('displays error message on fetch failure', async () => {
    fetch.mockRejectedValueOnce(new Error('Network error'));

    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });

  test('filters projects by status', async () => {
    const mockProjects = [
      { id: '1', name: 'Running Project', status: 'running', progress: 50 },
      { id: '2', name: 'Completed Project', status: 'completed', progress: 100 }
    ];

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockProjects
    });

    render(
      <ProjectProvider>
        <ProjectManager />
      </ProjectProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Running Project')).toBeInTheDocument();
    });

    const filterSelect = screen.getByRole('combobox', { name: /filter/i });
    fireEvent.change(filterSelect, { target: { value: 'completed' } });

    expect(screen.getByText('Completed Project')).toBeInTheDocument();
    expect(screen.queryByText('Running Project')).not.toBeInTheDocument();
  });
});
