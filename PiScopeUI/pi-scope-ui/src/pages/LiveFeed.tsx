import React from "react";
import { Container, Typography, Box } from "@mui/material";
import "../LiveFeed.css";

/**
 * LiveFeed Component
 * Renders a header and the live video feed from the backend.
 * The video feed is displayed in an <img> element whose source is the Flask endpoint.
 */
const LiveFeed: React.FC = () => {
  return (
    <Container maxWidth="md" className="live-feed-container">
      {/* Header */}
      <Typography variant="h4" component="h1" align="center" gutterBottom>
        Live feed from your device
      </Typography>
      
      {/* Video Feed */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <img 
          src="http://localhost:5001/video_feed"  // Make sure this URL matches your backend configuration
          alt="Live Feed" 
          className="live-feed-video"
        />
      </Box>
    </Container>
  );
};

export default LiveFeed;
