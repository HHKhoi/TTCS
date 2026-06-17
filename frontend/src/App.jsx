import { useState } from "react";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setFileName(file.name);
    }
  };

  const handleProcess = async () => {
    setLoading(true);
    setResult("");
    setExtractedText("");

    try {
      let response;
      if (mode === "text") {
        if (!text.trim()) {
          alert("Please enter some text!");
          setLoading(false);
          return;
        }
        response = await fetch("http://127.0.0.1:5000/api/detect/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text })
        });
      } else {
        if (!imageFile) {
          alert("Please upload an image!");
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append("image", imageFile);
        response = await fetch("http://127.0.0.1:5000/api/detect/image", {
          method: "POST",
          body: formData
        });
      }

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || "An error occurred");
      }

      const confidence = (data.score * 100).toFixed(1);
      setResult(`${data.label === "AI" ? "AI-generated" : "Human-written"} (${confidence}%)`);
      setExtractedText(data.extracted_text);

    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">AI Text Detection</h1>

      <div className="switch">
        <button
          className={`switch-btn ${mode === "text" ? "active" : ""}`}
          onClick={() => { setMode("text"); setResult(""); setExtractedText(""); }}
        >
          Paste Text
        </button>

        <button
          className={`switch-btn ${mode === "image" ? "active" : ""}`}
          onClick={() => { setMode("image"); setResult(""); setExtractedText(""); }}
        >
          Upload Image
        </button>
      </div>

      <div className="card">
        {mode === "text" ? (
          <textarea
            className="textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your text here..."
          />
        ) : (
          <div>
            <label className="upload-box">
              <input type="file" accept="image/*" onChange={handleImageUpload} hidden />
               Click to upload image
            </label>

            {fileName && (
              <div className="file-info">
                 {fileName}
                <button className="delete-btn" onClick={() => { setFileName(""); setImageFile(null); }}>
                  ×
                </button>
              </div>
            )}
          </div>
        )}

        <button className="main-btn" onClick={handleProcess} disabled={loading}>
          {loading ? "Processing..." : "Check AI"}
        </button>
      </div>

      {result && (
        <div className="result-card">
          <h3>Result</h3>
          <p style={{ fontSize: '1.2rem', fontWeight: 'bold', color: result.includes('AI') ? '#ef4444' : '#10b981' }}>
            {result}
          </p>
          {extractedText && (
            <div style={{ marginTop: '1rem', textAlign: 'left', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
              <strong>Extracted Text (OCR):</strong>
              <p style={{ fontSize: '0.9rem', marginTop: '5px', whiteSpace: 'pre-wrap' }}>{extractedText}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
