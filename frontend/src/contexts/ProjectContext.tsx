import React, { createContext, useContext, useState, ReactNode } from 'react';

interface Project {
  id: string;
  name: string;
  status: 'running' | 'paused' | 'completed' | 'error';
  progress: number;
  created_at: string;
  last_updated: string;
}

interface ProjectContextType {
  projects: Project[];
  activeProject: Project | null;
  setActiveProject: (project: Project | null) => void;
  addProject: (project: Project) => void;
  updateProject: (id: string, updates: Partial<Project>) => void;
  removeProject: (id: string) => void;
}

const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

interface ProjectProviderProps {
  children: ReactNode;
}

export const ProjectProvider: React.FC<ProjectProviderProps> = ({ children }) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);

  const addProject = (project: Project) => {
    setProjects(prev => [...prev, project]);
  };

  const updateProject = (id: string, updates: Partial<Project>) => {
    setProjects(prev =>
      prev.map(project =>
        project.id === id ? { ...project, ...updates } : project
      )
    );
  };

  const removeProject = (id: string) => {
    setProjects(prev => prev.filter(project => project.id !== id));
    if (activeProject?.id === id) {
      setActiveProject(null);
    }
  };

  const value: ProjectContextType = {
    projects,
    activeProject,
    setActiveProject,
    addProject,
    updateProject,
    removeProject,
  };

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
};

export const useProject = (): ProjectContextType => {
  const context = useContext(ProjectContext);
  if (context === undefined) {
    throw new Error('useProject must be used within a ProjectProvider');
  }
  return context;
};