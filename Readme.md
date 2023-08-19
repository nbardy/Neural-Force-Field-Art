## Neural Force Feld Art

This is an experiment of mine that [started years ago](https://github.com/nbardy/force-field-ml-art). My goal is to make art with neural
networks that optimizes movement of particles with neural networks.

This is a V2 that is a rewrite with a couple initialize goals:
 - [x] Supports multiple agents acting on separate sets
 - [ ] Supports modern transformers[In progress]
 - [ ] Supports incredibly fast rendering by keeping the state,learning, and drawing data all on the GPU[Almost Done, see quickDraw]
 - [ ] Supports 3D points[Done in a lot of the ML code, started in renderer]

A playground for experiments with TFJS Agents in the browser. 

The initial agent is set of points being moved around in a force field. This

Also impliments two libraries that were missing to make this work:
 * transformers.js - Modern transformer architectures for Typescript/Javascript
 * quickDraw - [Incredibly rendering via tfjs+webgl](https://github.com/nbardy/Neural-Force-Field-Art/tree/main/src/quickDraw)https://github.com/nbardy/Neural-Force-Field-Art/tree/main/src/quickDraw
