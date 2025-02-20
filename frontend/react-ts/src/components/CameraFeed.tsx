import React, { FC, useEffect } from "react";
import axios from "axios";

const fetchCamFeed = async () => {
    const response = await axios.get("http://localhost:5001/camera_web");
    console.log("Displaying Raw Cam feed")
}

useEffect(() => {
    fetchCamFeed();
});

const CameraFeed: FC = () => {
    //Currently retreiving the feed by using an iframe pointing to the Flask end
    //-point
    return (
        <div style={{ padding: '1rem' }}>
        <h1>Camera Feed</h1>
        <img 
            src="http://localhost:5001/camera_web" 
            alt="Raw Camera Feed" 
            style={{ width: '100%', maxWidth: '640px' }}
        />
        </div>
    )
}

export default CameraFeed; 