const API_BASE_URL = 'http://localhost:8000';

class APIService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Project endpoints
  async createProject(projectData) {
    return this.request('/projects', {
      method: 'POST',
      body: JSON.stringify(projectData),
    });
  }

  async getProjects() {
    return this.request('/projects');
  }

  async getProject(projectId) {
    return this.request(`/projects/${projectId}`);
  }

  async pauseProject(projectId) {
    return this.request(`/projects/${projectId}/pause`, {
      method: 'POST',
    });
  }

  async resumeProject(projectId) {
    return this.request(`/projects/${projectId}/resume`, {
      method: 'POST',
    });
  }

  // Task endpoints
  async createTask(projectId, taskData) {
    return this.request(`/projects/${projectId}/tasks`, {
      method: 'POST',
      body: JSON.stringify(taskData),
    });
  }

  // Agent endpoints
  async getAgents(projectId) {
    return this.request(`/projects/${projectId}/agents`);
  }

  async getAgent(projectId, agentId) {
    return this.request(`/projects/${projectId}/agents/${agentId}`);
  }

  // Metrics endpoints
  async getMetrics(projectId) {
    return this.request(`/projects/${projectId}/metrics`);
  }

  async getIterations(projectId) {
    return this.request(`/projects/${projectId}/iterations`);
  }

  // Memory endpoints
  async getErrors(projectId) {
    return this.request(`/projects/${projectId}/memory/errors`);
  }

  async searchMemory(projectId, query) {
    return this.request(`/projects/${projectId}/memory/search`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // Code endpoints
  async getCode(projectId, filePath) {
    return this.request(`/projects/${projectId}/code/${filePath}`);
  }

  // Checkpoint endpoints
  async createCheckpoint(projectId) {
    return this.request(`/projects/${projectId}/checkpoint`, {
      method: 'POST',
    });
  }

  async restoreCheckpoint(projectId, checkpointId) {
    return this.request(`/projects/${projectId}/restore/${checkpointId}`, {
      method: 'POST',
    });
  }

  // Health check
  async checkHealth() {
    return this.request('/health');
  }
}

export default new APIService();