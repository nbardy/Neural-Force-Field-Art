import React, { useState, useRef, useLayoutEffect } from "react";
import { createRoot } from "react-dom/client";
import { startLoop, GALLERY } from "./main";

const container = document.getElementById("app");
if (!container) throw new Error("Container not found");
const root = createRoot(container);
root.render(<App />);

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const [active, setActive] = useState(0);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (cleanupRef.current) cleanupRef.current();
    cleanupRef.current = startLoop(canvas, active);
    return () => {
      if (cleanupRef.current) cleanupRef.current();
      cleanupRef.current = null;
    };
  }, [active]);

  return (
    <>
      <canvas ref={canvasRef} id="myCanvas" />
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          display: "flex",
          gap: 6,
          padding: "8px 12px",
          background: "rgba(0,0,0,0.55)",
          zIndex: 100,
        }}
      >
        {GALLERY.map((piece, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            style={{
              padding: "6px 14px",
              border: i === active ? "1px solid #a070ff" : "1px solid #444",
              borderRadius: 4,
              background: i === active ? "rgba(100,50,200,0.4)" : "rgba(30,20,50,0.6)",
              color: i === active ? "#e0d0ff" : "#888",
              cursor: "pointer",
              fontSize: 13,
              fontFamily: "monospace",
            }}
          >
            {piece.name}
          </button>
        ))}
      </div>
    </>
  );
}
