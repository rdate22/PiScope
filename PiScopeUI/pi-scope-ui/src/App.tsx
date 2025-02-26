// App.tsx
import React, { useState, useRef, ChangeEvent } from 'react';
import './App.css';
import { NavBar } from './components/NavBar';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LiveFeed from './pages/LiveFeed'; // Import your LiveFeed page
import {
  Box,
  Button,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  IconButton,
  Tabs,
  Tab,
  TextField,
  Typography
} from '@mui/material';
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import axios from 'axios';

/**
 * Instructional slides for the landing page slideshow.
 */
const slides: string[] = [
  "Step 1: Scroll down and click 'Add User'.",
  "Step 2: Take 5-10 pictures of your face at different angles in good lighting.",
  "Step 3: Add your user's name.",
  "Step 4: Done! Your user will be added to your device.",
];

/**
 * CameraCapture component:
 * - Uses getUserMedia API to show a live camera preview.
 * - On clicking "Capture Photo", it takes a snapshot and passes it as a base64 data URL.
 */
interface CameraCaptureProps {
  onCapture: (imageDataUrl: string) => void;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  React.useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
    startCamera();

    // Cleanup: stop all video tracks when component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  // Capture a frame and pass it back as a data URL
  const handleCapture = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageDataUrl = canvas.toDataURL('image/jpeg');
        onCapture(imageDataUrl);
      }
    }
  };

  return (
    <Box sx={{ textAlign: 'center' }}>
      <video ref={videoRef} autoPlay style={{ width: '100%', maxWidth: '640px' }} />
      <Button variant="contained" color="primary" onClick={handleCapture} sx={{ mt: 1 }}>
        Capture Photo
      </Button>
    </Box>
  );
};

/**
 * AddUserDialog component:
 * - Renders a dialog for the user to either upload photos or capture them.
 * - Also includes a text field for entering the user's name.
 * - On confirmation, it creates a FormData payload and sends it via axios to the /add_user endpoint.
 */
interface AddUserDialogProps {
  open: boolean;
  onClose: () => void;
}

const AddUserDialog: React.FC<AddUserDialogProps> = ({ open, onClose }) => {
  const [tabValue, setTabValue] = useState(0); // 0 = upload, 1 = capture
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [userName, setUserName] = useState("");

  // Switch between "Upload Photos" and "Capture Photos" tabs
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle file input change
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUploadedFiles(e.target.files);
    }
  };

  // Add a captured image (data URL) to state
  const handleCapture = (imageDataUrl: string) => {
    setCapturedImages(prev => [...prev, imageDataUrl]);
  };

  // Utility: Convert a data URL to a Blob
  const dataURLtoBlob = (dataurl: string): Blob => {
    const arr = dataurl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch) throw new Error("Invalid data URL");
    const mime = mimeMatch[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while(n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  };

  /**
   * handleConfirm:
   * - Validates input and photos.
   * - Creates a FormData object and sends a POST request to /add_user.
   */
  const handleConfirm = async () => {
    if (!userName) {
      alert("Please enter a user name.");
      return;
    }
    if (tabValue === 0 && (!uploadedFiles || uploadedFiles.length < 5)) {
      alert("Please upload at least 5 photos.");
      return;
    }
    if (tabValue === 1 && capturedImages.length < 5) {
      alert("Please capture at least 5 photos.");
      return;
    }

    const formData = new FormData();
    formData.append('userName', userName);

    if (tabValue === 0 && uploadedFiles) {
      Array.from(uploadedFiles).forEach((file) => {
        formData.append('photos', file, file.name);
      });
    } else if (tabValue === 1) {
      capturedImages.forEach((dataUrl, index) => {
        const blob = dataURLtoBlob(dataUrl);
        formData.append('photos', blob, `capture-${index}.jpg`);
      });
    }

    try {
      const response = await axios.post("http://localhost:5001/add_user", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log("User added successfully:", response.data);
      alert("User added successfully!");
    } catch (error) {
      console.error("Error uploading user photos:", error);
      alert("Error adding user. Please try again.");
    }

    // Reset dialog state and close the dialog
    setUserName("");
    setUploadedFiles(null);
    setCapturedImages([]);
    setTabValue(0);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Add New User</DialogTitle>
      <DialogContent dividers>
        <TextField
          autoFocus
          margin="dense"
          label="User Name"
          type="text"
          fullWidth
          variant="outlined"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
        />
        <Tabs value={tabValue} onChange={handleTabChange} centered sx={{ my: 2 }}>
          <Tab label="Upload Photos" />
          <Tab label="Capture Photos" />
        </Tabs>
        {tabValue === 0 ? (
          <Box>
            <Typography variant="body1" gutterBottom>
              Please select 5-10 photos from your device. Accepted file types: .jpg, .jpeg, .png, .bmp, .tiff
            </Typography>
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.tiff"
              multiple
              onChange={handleFileChange}
            />
          </Box>
        ) : (
          <Box>
            <Typography variant="body1" gutterBottom>
              Use your camera to capture 5-10 photos.
            </Typography>
            <CameraCapture onCapture={handleCapture} />
            <Grid container spacing={1} sx={{ mt: 1 }}>
              {capturedImages.map((img, idx) => (
                <Grid item xs={4} key={idx}>
                  <img src={img} alt={`capture-${idx}`} style={{ width: '100%' }} />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="secondary">
          Cancel
        </Button>
        <Button onClick={handleConfirm} variant="contained" color="primary">
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );
};

/**
 * LandingPage component encapsulates your current landing page UI.
 * This component will be rendered at the "/" route.
 */
const LandingPage: React.FC = () => {
  const [currentSlide, setCurrentSlide] = useState<number>(0);
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);

  const handlePrevSlide = () => {
    setCurrentSlide((prev) => (prev === 0 ? slides.length - 1 : prev - 1));
  };

  const handleNextSlide = () => {
    setCurrentSlide((prev) => (prev === slides.length - 1 ? 0 : prev + 1));
  };

  const handleAddUserClick = () => {
    setDialogOpen(true);
  };

  const handleDialogClose = () => {
    setDialogOpen(false);
  };

  return (
    <Box sx={{ my: 4, textAlign: 'center' }}>
      <Typography variant="h3" component="h1" gutterBottom sx={{ fontFamily: 'Georgia, serif' }}>
        Welcome to SentriLock
      </Typography>
      <Typography variant="h6" component="p" gutterBottom>
        Follow these steps to add a new user to your SentriLock device:
      </Typography>

      {/* Slideshow with manual navigation */}
      <Card sx={{ maxWidth: 640, margin: '0 auto', my: 4, padding: 2 }}>
        <CardContent>
          <Typography variant="h5" component="div" sx={{ fontFamily: 'Georgia, serif', fontStyle: 'italic' }}>
            {slides[currentSlide]}
          </Typography>
        </CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', px: 2, pb: 2 }}>
          <IconButton onClick={handlePrevSlide}>
            <ArrowBackIosIcon />
          </IconButton>
          <IconButton onClick={handleNextSlide}>
            <ArrowForwardIosIcon />
          </IconButton>
        </Box>
      </Card>

      <Button variant="contained" color="primary" onClick={handleAddUserClick}>
        Add User
      </Button>

      <AddUserDialog open={dialogOpen} onClose={handleDialogClose} />
    </Box>
  );
};

/**
 * Main App component:
 * - Wraps the application in a Router.
 * - Defines two routes:
 *    "/" renders the LandingPage.
 *    "/live-feed" renders the LiveFeed page.
 */
function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/live-feed" element={<LiveFeed />} />
      </Routes>
    </Router>
  );
}

export default App;
