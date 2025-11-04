import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import toast from 'react-hot-toast';

const WebSocketContext = createContext(null);

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [messageQueue, setMessageQueue] = useState([]);

  useEffect(() => {
    // Connect to WebSocket server
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setSocket(ws);
      toast.success('Connected to server');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        setLastMessage(data);
        setMessageQueue(prev => [...prev, data]);

        // Handle different message types
        switch (data.type) {
          case 'project_update':
            toast.info(`Project ${data.project_name}: ${data.status}`);
            break;
          case 'agent_update':
            console.log(`Agent ${data.agent}: ${data.status}`);
            break;
          case 'error':
            toast.error(`Error: ${data.message}`);
            break;
          case 'milestone':
            toast.success(`Milestone reached: ${data.milestone}`);
            break;
          case 'iteration':
            console.log(`Iteration ${data.iteration}: ${data.status}`);
            break;
          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setSocket(null);
      toast.warning('Disconnected from server');

      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        console.log('Attempting to reconnect...');
      }, 3000);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, []);

  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    console.error('WebSocket is not connected');
    return false;
  }, [socket]);

  const clearMessageQueue = useCallback(() => {
    setMessageQueue([]);
  }, []);

  const value = {
    socket,
    isConnected,
    lastMessage,
    messageQueue,
    sendMessage,
    clearMessageQueue
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};