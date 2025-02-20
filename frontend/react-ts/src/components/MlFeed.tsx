import axios from "axios";
import React, { FC, useEffect } from "react";

const fetchMLFeed = async () => {
    const response = await axios.get("http://localhost:5001/camera_web");
    console.log("Displaying video feed")
}

useEffect(() => {
    fetchMLFeed();
});

const MlFeed: FC = () => {
    return (
        <div style = {{ padding: '1rem' }}>
            <h1>Camera feed with Bounding boxes</h1>
            <img
                src="http://localhost:5001/ml_feed" 
                alt="ML Camera Feed with Bounding boxes" 
                style={{ width: '100%', maxWidth: '640px' }}
            />
        </div>
    );
}

export default MlFeed;