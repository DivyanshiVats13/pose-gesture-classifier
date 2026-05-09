import { useEffect, useRef, useState } from "react";

export default function App() {
  const [gesture, setGesture]     = useState("—");
  const [conf, setConf]           = useState(0);
  const [connected, setConnected] = useState(false);
  const [error, setError]         = useState(null);
  const imgRef     = useRef(null);
  const wsRef      = useRef(null);
  const historyRef = useRef([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen  = () => { setConnected(true); setError(null); };
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setError("Cannot connect to backend. Make sure uvicorn is running.");

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      historyRef.current.push(data.gesture);
      if (historyRef.current.length > 5) historyRef.current.shift();
      const mode = [...historyRef.current].sort((a,b) =>
        historyRef.current.filter(v=>v===b).length - historyRef.current.filter(v=>v===a).length
      )[0];
      setGesture(mode);
      setConf(data.confidence);
      if (imgRef.current) {
        imgRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
    };

    return () => ws.close();
  }, []);

  const confColor = conf > 0.85 ? "#1D9E75" : conf > 0.6 ? "#BA7517" : "#E24B4A";

  const gestureEmoji = {
    thumbs_up:   "👍",
    thumbs_down: "👎",
    open_palm:   "🖐",
    fist:        "✊",
    point:       "☝️",
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f0f0f", color: "#fff",
                  fontFamily: "sans-serif", padding: "24px" }}>

      <div style={{ maxWidth: 760, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 4px" }}>
            Real-time Gesture Classifier
          </h1>
          <p style={{ fontSize: 13, color: "#888", margin: 0 }}>
            MediaPipe · Random Forest · FastAPI · React
          </p>
        </div>

        {/* Status bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 8,
                      marginBottom: 16 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%",
                         background: connected ? "#1D9E75" : "#E24B4A",
                         display: "inline-block" }} />
          <span style={{ fontSize: 13, color: "#aaa" }}>
            {connected ? "Connected to backend" : "Disconnected"}
          </span>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#2a0a0a", border: "1px solid #5a1a1a",
                        borderRadius: 8, padding: "12px 16px",
                        fontSize: 13, color: "#f09595", marginBottom: 16 }}>
            {error}
          </div>
        )}

        {/* Camera feed */}
        <img ref={imgRef} alt="live webcam feed"
             style={{ width: "100%", borderRadius: 12,
                      border: "1px solid #222", minHeight: 200,
                      background: "#1a1a1a", display: "block" }} />

        {/* Gesture result */}
        <div style={{ marginTop: 16, display: "grid",
                      gridTemplateColumns: "1fr 1fr", gap: 12 }}>

          {/* Gesture label */}
          <div style={{ background: "#1a1a1a", border: "1px solid #2a2a2a",
                        borderRadius: 12, padding: "16px 20px" }}>
            <p style={{ fontSize: 12, color: "#666", margin: "0 0 6px" }}>
              Detected gesture
            </p>
            <p style={{ fontSize: 28, fontWeight: 600, margin: 0,
                        textTransform: "capitalize", letterSpacing: "-0.5px" }}>
              {gestureEmoji[gesture] || ""} {gesture}
            </p>
          </div>

          {/* Confidence */}
          <div style={{ background: "#1a1a1a", border: "1px solid #2a2a2a",
                        borderRadius: 12, padding: "16px 20px" }}>
            <p style={{ fontSize: 12, color: "#666", margin: "0 0 6px" }}>
              Confidence
            </p>
            <p style={{ fontSize: 28, fontWeight: 600, margin: "0 0 10px",
                        color: confColor }}>
              {Math.round(conf * 100)}%
            </p>
            <div style={{ height: 6, background: "#2a2a2a", borderRadius: 99 }}>
              <div style={{ height: "100%", borderRadius: 99,
                            width: `${conf * 100}%`, background: confColor,
                            transition: "width 0.15s ease" }} />
            </div>
          </div>
        </div>

        {/* Gesture guide */}
        <div style={{ marginTop: 12, background: "#1a1a1a",
                      border: "1px solid #2a2a2a", borderRadius: 12,
                      padding: "14px 20px" }}>
          <p style={{ fontSize: 12, color: "#666", margin: "0 0 10px" }}>
            Supported gestures
          </p>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            {["👍 thumbs up", "👎 thumbs down", "🖐 open palm",
              "✊ fist", "☝️ point"].map(g => (
              <span key={g} style={{ fontSize: 13, color: "#aaa" }}>{g}</span>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
