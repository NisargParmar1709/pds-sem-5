# Video Sentiment AI 🎭

A full-stack artificial intelligence web application that analyzes video files to extract and detect human speech, emotions, and sentiments in real-time. 

---

## 🏗️ Architecture

This project is separated into clean, decoupled components to ensure a production-ready deployment:

- **Frontend (`/frontend`)**: A sleek, dark-mode React UI built with Vite. It features drag-and-drop video uploading and beautifully animated metric visualizations.
- **Backend (`/backend`)**: A FastAPI Python server that loads the PyTorch AI model into memory and exposes a highly efficient `POST /analyze` endpoint.
- **Training (`/AI model`)**: The original datasets, virtual environments, TensorBoard logs, and PyTorch training scripts used to build the model *(Note: This folder is excluded from version control to save space).*

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your machine:
- **Node.js** (v16 or higher)
- **Python** (v3.9 or higher)
- **FFmpeg** (Required by the backend for audio extraction)

---

## 🚀 Installation & Setup

### 1. Backend Setup
1. Open your terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. *Note: Ensure your trained `model.pt` file is placed inside the `backend/model/` directory.*

### 2. Frontend Setup
1. Open a new terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
   ```bash
   npm install
   ```

---

## 💻 How to Run the App Locally

To use the application, you need to run both the backend server and the frontend interface simultaneously.

### Start the Backend Server:
From the `backend` folder with your virtual environment activated, run:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`*

### Start the Frontend Server:
From the `frontend` folder, run:
```bash
npm run dev
```
*The UI will be available at `http://localhost:5173`*

Open the frontend URL in your browser, upload an MP4/AVI file, and click **Analyze Video**!

---

## 📊 Viewing Training Data (TensorBoard)

If you are developing locally and want to view the training charts (loss, accuracy, etc.), you can boot up TensorBoard.

1. Open a terminal and navigate to your training folder:
   ```bash
   cd "AI model/training"
   ```
2. Activate your original training virtual environment.
3. Run the TensorBoard module pointing to your runs folder:
   ```bash
   python -m tensorboard.main --logdir="runs"
   ```
4. Open the provided link (usually `http://localhost:6006`) in your browser to view your visualized AI metrics.
