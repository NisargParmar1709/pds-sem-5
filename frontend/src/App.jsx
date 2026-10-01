import { useState, useRef } from 'react';
import { UploadCloud, Video, AlertCircle, FileVideo, Activity, BarChart2 } from 'lucide-react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile) => {
    setError(null);
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
      setError('Please upload a valid video file (.mp4, .avi, .mov, .mkv)');
      return;
    }
    setFile(selectedFile);
    setResult(null);
  };

  const analyzeVideo = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || 'Analysis failed on server.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the server.');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Video Sentiment AI</h1>
        <p>Upload a video to analyze speech emotions and sentiments in real-time.</p>
      </header>

      <main>
        <div className="glass-panel">
          <div 
            className={`upload-area ${isDragging ? 'active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileInput} 
              className="file-input" 
              accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
            />
            
            {file ? (
              <div>
                <FileVideo className="upload-icon" style={{margin: '0 auto'}} />
                <h3 style={{marginTop: '1rem', fontSize: '1.25rem'}}>{file.name}</h3>
                <p style={{color: 'var(--text-muted)'}}>{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            ) : (
              <div>
                <UploadCloud className="upload-icon" style={{margin: '0 auto'}} />
                <h3 style={{marginTop: '1rem', fontSize: '1.25rem'}}>Drag & Drop Video Here</h3>
                <p style={{color: 'var(--text-muted)'}}>or click to browse files</p>
              </div>
            )}
          </div>

          {error && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.1)', 
              color: '#ef4444', 
              padding: '1rem', 
              borderRadius: '0.5rem',
              marginTop: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          <div style={{marginTop: '2rem', textAlign: 'center'}}>
            <button 
              className="btn" 
              onClick={analyzeVideo} 
              disabled={!file || loading}
            >
              <Video size={20} />
              {loading ? 'Analyzing...' : 'Analyze Video'}
            </button>
          </div>
        </div>

        {loading && (
          <div className="progress-container">
            <Activity className="spinner" />
            <h3 style={{fontSize: '1.25rem', marginBottom: '0.5rem'}}>Processing Video...</h3>
            <p style={{color: 'var(--text-muted)'}}>Extracting features, transcribing speech, and analyzing emotions. This may take a moment.</p>
          </div>
        )}

        {result && result.utterances && (
          <div className="glass-panel" style={{marginTop: '2rem'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem'}}>
              <BarChart2 size={24} color="var(--primary)" />
              <h2 style={{fontSize: '1.5rem', fontWeight: '700'}}>Analysis Results</h2>
            </div>

            <div className="results-grid">
              {result.utterances.map((utt, index) => (
                <div key={index} className="utterance-card">
                  <div className="utterance-header">
                    <p className="utterance-text">"{utt.text}"</p>
                    <span className="utterance-time">
                      {formatTime(utt.start_time)} - {formatTime(utt.end_time)}
                    </span>
                  </div>

                  <div className="metrics-container">
                    <div className="metric-section">
                      <h4>Top Emotions</h4>
                      {utt.emotions.map((emo, i) => (
                        <div key={i} className="metric-bar">
                          <div className="metric-label">
                            <span style={{textTransform: 'capitalize'}}>{emo.label}</span>
                            <span>{(emo.confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div className="metric-track">
                            <div 
                              className="metric-fill emotion" 
                              style={{width: `${emo.confidence * 100}%`}}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="metric-section">
                      <h4>Sentiments</h4>
                      {utt.sentiments.map((sent, i) => (
                        <div key={i} className="metric-bar">
                          <div className="metric-label">
                            <span style={{textTransform: 'capitalize'}}>{sent.label}</span>
                            <span>{(sent.confidence * 100).toFixed(0)}%</span>
                          </div>
                          <div className="metric-track">
                            <div 
                              className="metric-fill sentiment" 
                              style={{width: `${sent.confidence * 100}%`}}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
              {result.utterances.length === 0 && (
                <div style={{textAlign: 'center', padding: '2rem', color: 'var(--text-muted)'}}>
                  No speech detected in the video.
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
