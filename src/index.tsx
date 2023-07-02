import React from "react";

import { mainLoop } from "./main";
import { useLayoutEffect } from "react";

import { createRoot } from "react-dom/client";
import { App } from "./App";

const container = document.getElementById("app");
const root = createRoot(container)
root.render(<App />);

export default function App() {
  // canvas ref
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;

    mainLoop(canvas);
  }, []);

  return (
    <>
      <canvas ref={canvasRef} id="myCanvas"></canvas>
    </>
  );
}
