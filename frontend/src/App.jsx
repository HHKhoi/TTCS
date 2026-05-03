import { useState } from "react";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
    }
  };

  const handleProcess = () => {
    setLoading(true);

    setTimeout(() => {
      const isAI = Math.random() > 0.5;
      const confidence = (Math.random() * 20 + 80).toFixed(2);

      setResult(`${isAI ? "AI-generated" : "Human-written"} (${confidence}%)`);

      // reset sau khi check
      setText("");
      setFileName("");

      setLoading(false);
    }, 1000);
  };

  return (
    <div className="container">
      <h1 className="title">AI Text Detection</h1>

      {/* MODE SWITCH */}
      <div className="switch">
        <button
          className={`switch-btn ${mode === "text" ? "active" : ""}`}
          onClick={() => setMode("text")}
        >
          Paste Text
        </button>

        <button
          className={`switch-btn ${mode === "image" ? "active" : ""}`}
          onClick={() => setMode("image")}
        >
          Upload Image
        </button>
      </div>

      {/* INPUT */}
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
              <input type="file" onChange={handleImageUpload} hidden />
              📁 Click to upload image
            </label>

            {fileName && (
              <div className="file-info">
                📄 {fileName}
                <button className="delete-btn" onClick={() => setFileName("")}>
                  ×
                </button>
              </div>
            )}
          </div>
        )}

        <button className="main-btn" onClick={handleProcess}>
          {loading ? "Processing..." : "Check AI"}
        </button>
      </div>

      {/* RESULT */}
      <div className="result-card">
        <h3>Result</h3>
        <p>{result}</p>
      </div>
    </div>
  );
}
