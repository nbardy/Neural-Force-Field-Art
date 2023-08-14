## Quick Draw

An artistic generative art library that runs entirely on the GPU

Why?

1. Drawing counts in the 100,000s should be available to generative artists who know only
   drawTriangle, drawCircle.

   Right now generative artists need to be able to write shaders to draw with WebGL. This will
   provide access to that with a high level API familiar to them.

2. I want to be able to render from large tfjs tensors that are holding state and
   being computed on by tfjs. Reusing another API would require copying the data to CPU
   or doing compute on the CPU. Computing the geometry on the GPU is much faster.

   Provides one simple function for passing data back and forth between tfjs
   and into shaders, which allows custom rendering build on top of tfjs tensors
   for those want no limitations.

How?

WebGL has no geometry shaders so we do the geometry calculations on GPU with tfjs.
Other shader libraries require you to create CPU object and do geometry calculations
on the CPU which are slow for computation and causes a lot of data transfer between CPU
and GPU.

We also accept tfjs tensors directly as input to the drawing functions. This means you can
compute your update code with tfjs and draw the results directly without copying to CPU. Instead
of onlying using the GPU for drawing you can use it for computation as well. We want
our scenes to both render fast and update fast.

Consists of:

1.  One helper function (drawTWGL that draws TWGIL with default gl.TRIANGLES) or another shape.
2.  Artistic Layer that provides drawing oriented APIs instead of GPU oriented APIs
    e.g. Position, Direction, etc... over triangle vertices

Starting as a simple wrapper over twgl.js

Example wave

// range
const x = tf.linspace(-1, 1, 100);
let y = tf.sin(x);

// loop and draw updating positions with sin wave
while (true) {
const t = tf.scalar(Date.now() \* 0.001);

// move up and down
y = y + tf.sin(x + t);

// draw circle
drawCircles(canvas, x, y, 0.01);
}
