import React, { useState, useEffect } from 'react'
import { useProject } from '../contexts/ProjectContext'

function ProjectManager() {
  const [projects, setProjects] = useState([])
  const [isCreating, setIsCreating] = useState(false)
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    requirements: []
  })

  useEffect(() => {
    fetchProjects()
  }, [])

  const fetchProjects = async () => {
    try {
      const response = await fetch('http://localhost:8000/projects')
      const data = await response.json()
      setProjects(data)
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    }
  }

  const createProject = async () => {
    try {
      const response = await fetch('http://localhost:8000/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newProject)
      })

      if (response.ok) {
        setIsCreating(false)
        setNewProject({ name: '', description: '', requirements: [] })
        fetchProjects()
      }
    } catch (error) {
      console.error('Failed to create project:', error)
    }
  }

  return (
    <div className="project-manager">
      <div className="project-header">
        <h2>Projects</h2>
        <button onClick={() => setIsCreating(true)} className="btn-primary">
          + New Project
        </button>
      </div>

      {isCreating && (
        <div className="create-project-form">
          <h3>Create New Project</h3>
          <input
            type="text"
            placeholder="Project Name"
            value={newProject.name}
            onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
            className="input"
          />
          <textarea
            placeholder="Description"
            value={newProject.description}
            onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
            className="textarea"
            rows={4}
          />
          <div className="form-actions">
            <button onClick={createProject} className="btn-primary">Create</button>
            <button onClick={() => setIsCreating(false)} className="btn-secondary">Cancel</button>
          </div>
        </div>
      )}

      <div className="projects-list">
        {projects.map(project => (
          <div key={project.id} className="project-card">
            <h3>{project.name}</h3>
            <div className="project-meta">
              <span className="badge">{project.state}</span>
              <span>Iteration {project.iteration}</span>
            </div>
            <div className="project-metrics">
              {project.metrics && (
                <>
                  <div className="metric">
                    <span>Coverage:</span>
                    <span>{project.metrics.test_coverage?.toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <span>Performance:</span>
                    <span>{project.metrics.performance_score?.toFixed(1)}%</span>
                  </div>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      <style jsx>{`
        .project-manager {
          padding: 20px;
        }
        .project-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        .create-project-form {
          background: #1e1e1e;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
        }
        .input, .textarea {
          width: 100%;
          padding: 10px;
          margin: 10px 0;
          background: #2d2d2d;
          border: 1px solid #444;
          border-radius: 4px;
          color: #fff;
        }
        .form-actions {
          display: flex;
          gap: 10px;
          margin-top: 10px;
        }
        .btn-primary, .btn-secondary {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        .btn-primary {
          background: #007bff;
          color: white;
        }
        .btn-secondary {
          background: #6c757d;
          color: white;
        }
        .projects-list {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 20px;
        }
        .project-card {
          background: #1e1e1e;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #333;
        }
        .project-meta {
          display: flex;
          gap: 10px;
          margin: 10px 0;
        }
        .badge {
          background: #007bff;
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 12px;
        }
        .project-metrics {
          margin-top: 10px;
        }
        .metric {
          display: flex;
          justify-content: space-between;
          padding: 5px 0;
        }
      `}</style>
    </div>
  )
}

export default ProjectManager
