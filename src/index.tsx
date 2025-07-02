import React from "react";
import { useState, useEffect } from "react";


import { createRoot } from "react-dom/client";
import styled from 'styled-components';


const container = document.getElementById("app");
if (!container) throw new Error("Container not found");
const root = createRoot(container);

root.render(<App />);



export default function App() {
  const currentScene = useSceneJourney();

  return (
    <div id="main">
      {currentScene}
    </div>
  );
}


const Scene1 = (props: any) => {
  return (
    <>
      <div id="scene-1-text">Scene 1</div>
    </>
  )
}

const Scene2 = (props: any) => {
  return (
    <>
      <div id="scene-2-text">Scene 2</div>
    </>
  )
}

export const Scene3 = (props: any) => {
  return (
    <>
      <div id="scene-3-text">Scene 3</div>
    </>
  )
}

export const Scene4 = (props: any) => {
  return (
    <>
      <div id="scene-4-text">Scene 4</div>
    </>
  )
}

export const Scene5 = (props: any) => {
  return (
    <>
      <div id="scene-5-text">Scene 5</div>
    </>
  )
}

export const Scene6 = (props: any) => {
  return (
    <>
      <div id="scene-6-text">Scene 6</div>
    </>
  )
}

export const Scene7 = (props: any) => {
  return (
    <>
      <div id="scene-7-text">Scene 7</div>
    </>
  )
}

// const scenes = [
//   { time: 0, scene: 1, component: Scene1 },
//   { time: 60, scene: 2, component: Scene2 },
//   { time: 120, scene: 3, component: Scene3 },
//   { time: 180, scene: 4, component: Scene4 },
//   { time: 240, scene: 5, component: Scene5 },
//   { time: 300, scene: 6, component: Scene6 },
//   { time: 360, scene: 7, component: Scene7 },
// ]

const scenes = [
  { time: 0, scene: 1, component: Scene1 },
  { time: 6, scene: 2, component: Scene2 },
  { time: 8, scene: 3, component: Scene3 },
  { time: 10, scene: 4, component: Scene4 },
  { time: 12, scene: 5, component: Scene5 },
  { time: 23, scene: 6, component: Scene6 },
]


// This should return the proper scene component based on the time
const useSceneJourney = () => {
  const [currentScene, setCurrentScene] = useState(React.createElement(Scene1));
  
  useEffect(() => {
    // Get the initial time
    const initialTime = Math.floor(Date.now() / 1000); // time in seconds

    // Array to hold all setTimeout IDs for cleanup
    const timeoutIds: number[] = [];

    scenes.forEach((sceneInfo) => {
      // Calculate the delay for the setTimeout
      const delay = (sceneInfo.time - (Math.floor(Date.now() / 1000) - initialTime)) * 1000;

      // Set a timeout to update to this scene
      if (delay >= 0) { // [Change] Only set timeout if delay is non-negative
        const id = setTimeout(() => {
          setCurrentScene(React.createElement(sceneInfo.component));
        }, delay);
        timeoutIds.push(id);
      }
    });

    // Cleanup function to clear all the timeouts
    return () => timeoutIds.forEach((id) => clearTimeout(id));
  }, []);  // Empty dependency array to run the effect only once
  
  return currentScene;
};