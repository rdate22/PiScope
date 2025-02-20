import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import CameraFeed from './components/CameraFeed';
import MlFeed from './components/MlFeed';

const App: React.FC = () => {
  return (
    <Router>
      <nav style={{ padding: '1rem', backgroundColor: '#f0f0f0' }}>
        <Link to="/rawCam" style={{ marginRight: '1rem' }}>Camera Feed</Link>
        <Link to="/mlCam">ML Feed</Link>
      </nav>
      <Routes>
        <Route path="/rawCam" element={<CameraFeed />} />
        <Route path="/mlCam" element={<MlFeed />} />
        <Route path="/" element={<CameraFeed />} />
      </Routes>
    </Router>
  );
};

export default App;