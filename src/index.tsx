import React from "react";

import { startLoop } from "./main";
import { useLayoutEffect } from "react";

import { createRoot } from "react-dom/client";

const container = document.getElementById("app");
if (!container) throw new Error("Container not found");
const root = createRoot(container);

root.render(<App />);

export default function App() {
  // canvas ref
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    // throw error if canvas is not supported
    if (!canvas) throw new Error("Canvas not supported");

    startLoop(canvas);
  }, []);

  return (
    <>
      <canvas ref={canvasRef} id="myCanvas"></canvas>
    </>
  );
}
