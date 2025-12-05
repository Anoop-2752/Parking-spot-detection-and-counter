# Parking-spot-detection-and-counter
##  Project Overview

The goal of this project is to identify:

- Empty parking spaces  
- Occupied parking spaces  
- Total vehicles in the parking area  
- Real-time occupancy updates  

It works with:

- CCTV footage  
- Recorded videos  
- Live camera streams  

The system highlights each parking slot and displays live counters for occupied vs available spaces.


---

## 📷 How It Works

### **1. Image Preprocessing**
- Convert frame to grayscale  
- Apply Gaussian blur  
- Adaptive threshold  
- Morphological operations for noise removal  

### **2. Parking Slot ROI Mapping**
Each slot is defined using coordinate points (manually or auto-detected).  
These ROIs are checked frame-by-frame.

### **3. Occupancy Detection Logic**
For each slot ROI:
- Crop the region  
- Count non-zero pixels / edge density  
- If pixel count > threshold → **occupied**  
- Else → **empty**

### **4. Vehicle Counting**
Uses:
- Contour detection  
- Background subtraction OR  
- YOLO (optional) for more accuracy  

---

## 🧠 Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **(Optional)** YOLO / TensorFlow / PyTorch

---
