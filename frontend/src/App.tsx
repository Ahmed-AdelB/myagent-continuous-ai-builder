import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { ProjectProvider } from './contexts/ProjectContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  return (
    <Router>
      <ProjectProvider>
        <WebSocketProvider>
          <div className="App">
            <Routes>
              <Route path="/" element={<Dashboard />} />
            </Routes>
            <Toaster position="top-right" />
          </div>
        </WebSocketProvider>
      </ProjectProvider>
    </Router>
  );
}

export default App;