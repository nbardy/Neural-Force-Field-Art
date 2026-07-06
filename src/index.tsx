import React, { useState, useRef, useLayoutEffect } from "react";
import { createRoot } from "react-dom/client";
import { startLoop, GALLERY, type LoopHandle } from "./main";
import type { HelmholtzField } from "./core/field/helmholtz";

const container = document.getElementById("app");
if (!container) throw new Error("Container not found");
const root = createRoot(container);
root.render(<App />);

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  // Live handle to the active field (null for legacy MLP pieces). The alpha
  // slider mutates `field.alpha` in place — no React re-render needed.
  const fieldRef = useRef<HelmholtzField | null>(null);
  const handleRef = useRef<LoopHandle | null>(null);
  const [active, setActive] = useState(0);
  const [alpha, setAlpha] = useState(0.7);
  const [particles, setParticles] = useState(1000);
  const [samples, setSamples] = useState(256);

  // Only field-based pieces (createField) expose the order↔chaos alpha knob.
  const hasField = !!GALLERY[active].createField;

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (cleanupRef.current) cleanupRef.current();
    cleanupRef.current = startLoop(canvas, active, (h) => {
      handleRef.current = h;
      fieldRef.current = h.field;
      if (h.field) setAlpha(h.field.alpha);
      setParticles(h.getParticleCount());
      setSamples(h.getSampleRate());
    });
    return () => {
      if (cleanupRef.current) cleanupRef.current();
      cleanupRef.current = null;
      fieldRef.current = null;
    };
  }, [active]);

  return (
    <>
      <canvas ref={canvasRef} id="myCanvas" />
      <div
        style={{
          position: "fixed",
          bottom: 42,
          left: 0,
          right: 0,
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          background: "rgba(0,0,0,0.55)",
          color: "#8fbcff",
          fontFamily: "monospace",
          fontSize: 12,
          zIndex: 100,
        }}
      >
        <span style={{ width: 60 }}>particles</span>
        <input
          type="range"
          min={200}
          max={100000}
          step={200}
          value={particles}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            const v = parseInt(e.target.value, 10);
            setParticles(v);
            handleRef.current?.setParticleCount(v);
          }}
          style={{ flex: 1, maxWidth: 260, accentColor: "#5b8cff" }}
        />
        <span style={{ width: 60, textAlign: "right" }}>
          {particles.toLocaleString()}
        </span>
        <span style={{ width: 52, marginLeft: 18 }}>samples</span>
        <input
          type="range"
          min={16}
          max={1024}
          step={16}
          value={samples}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            const v = parseInt(e.target.value, 10);
            setSamples(v);
            handleRef.current?.setSampleRate(v);
          }}
          style={{ flex: 1, maxWidth: 180, accentColor: "#5b8cff" }}
        />
        <span style={{ width: 36, textAlign: "right" }}>{samples}</span>
      </div>
      {hasField && (
        <div
          style={{
            position: "fixed",
            bottom: 82,
            left: 0,
            right: 0,
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 12px",
            background: "rgba(0,0,0,0.55)",
            color: "#c0a0ff",
            fontFamily: "monospace",
            fontSize: 12,
            zIndex: 100,
          }}
        >
          <span>order</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={alpha}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              const v = parseFloat(e.target.value);
              setAlpha(v);
              // Mutate the live field directly — the render loop reads it next frame.
              if (fieldRef.current) fieldRef.current.alpha = v;
            }}
            style={{ flex: 1, maxWidth: 320, accentColor: "#a070ff" }}
          />
          <span>chaos</span>
          <span style={{ width: 40, textAlign: "right" }}>α {alpha.toFixed(2)}</span>
        </div>
      )}
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
