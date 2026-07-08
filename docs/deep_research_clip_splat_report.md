# Deep Research for a WebGPU CLIP-Guided Gaussian Splat Art Optimizer

## Bottom line

Your current system is already unusually well aligned with the most relevant literature: explicit differentiable rendering, shared multi-view optimization, and an efficient CLIP-family encoder are exactly the ingredients that made early CLIP-only text-to-3D work possible before diffusion-based score distillation took over. The most practically useful next gains are **not** “switch to diffusion,” but rather: **add Dream Fields–style opacity and background regularization, add Mip-Splatting–style anti-aliasing/frequency control, and move from purely volumetric 3D Gaussians toward a surface-biased hybrid representation** instead of a hard immediate jump to pure 2DGS surfels. Dream Fields directly shows that random background augmentations plus a transparency regularizer improve coherent object formation, and that broad pose sampling avoids flat billboard geometry; 2DGS and Gaussian Surfels show why flattened surface elements improve geometry and view consistency; Mip-Splatting shows why unconstrained Gaussian scale degenerates into aliasing-prone high-frequency artifacts. citeturn23view0turn17view0turn20search0turn12view4turn42view0

For your **browser/WebGPU** constraint, the best medium-term architecture is a **hybrid**: keep 3D Gaussians for early coarse object formation and stable visibility, then progressively flatten stable, high-opacity splats toward surfels or 2DGS-style disks, adding normal/depth regularization once meaningful surfaces emerge. That recommendation is an inference, but it is grounded in the literature: 3DGS is strong as a general explicit optimization substrate, while 2DGS, Gaussian Surfels, SuGaR, and GOF all improve surface quality by biasing the representation toward actual surfaces rather than volumetric “clouds.” citeturn21search17turn17view0turn20search0turn12view1turn7search4

On backgrounds, **do not keep black fixed for the whole optimization**. Dream Fields explicitly found that training against a simple fixed white or black background lets the model diffuse opacity into the background, while random backgrounds plus transmittance regularization produce sharper, more coherent objects. For your art direction, the right compromise is a **dark-background curriculum**: mostly black or dark procedural backgrounds during early and middle training, with a final short black-only fine-tune if you want the finished object to “read” on black. citeturn23view0

On prompts, **keep one shared base prompt across views**, and only add **weak directional wording** for coarse azimuth bins such as front, side, and back when the concept has a canonical orientation. DreamFusion found view-dependent prompting beneficial for geometry, but its own ablation also shows geometry is still fragile and improved further only when coupled with lighting and textureless renders; DreamView likewise treats the problem as balancing an overall prompt with view-specific guidance rather than replacing the overall prompt entirely. In a CLIP-only browser optimizer, nine bespoke prompts are more likely to inject semantic inconsistency than to help. citeturn25view0turn25view1turn31search8turn31search3

On CLIP batching, **yes eventually, but not first**. MobileCLIP is relevant because it is explicitly optimized for runtime efficiency and the official code path uses standard batched image/text encoding APIs, but the higher-value immediate wins are reducing pathological gradients and background cheating first. That said, WebGPU dispatch overhead is real at batch size 1, and a recent cross-platform characterization shows that dispatch-heavy ML pipelines suffer materially from per-operation overhead and can benefit from reducing dispatch count or amortizing it through more work per submission. So if profiling shows your MobileCLIP forward/backward dominates step time, micro-batching multiple views is probably worth doing after the regularizers land. citeturn15view0turn15view1turn33view0

## Ranked paper and repo table

| Rank | Paper or repo | Venue and year | Why it matters for your app | Classification | Primary source |
|---|---|---|---|---|---|
| 1 | **Dream Fields** | CVPR 2022 | The single most relevant CLIP-only 3D paper for your setup: random pose sampling, transmittance regularization, scene bounds, and background augmentation are all directly portable to explicit splats. citeturn23view0 | **directly implementable now** | citeturn8search7turn23view0 |
| 2 | **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** | SIGGRAPH 2024 | Strongest case for surface-biased splats: 2D oriented disks, perspective-correct ray–splat intersection, depth distortion, and normal consistency are exactly the ingredients most likely to improve multi-view coherence and real geometry. citeturn17view0turn42view1 | **directly implementable now** | citeturn17view0turn20search2 |
| 3 | **Mip-Splatting** | CVPR 2024 | Most practical anti-alias/frequency paper for your renderer. It directly addresses degenerate small Gaussians, zoom artifacts, and the need for a better screen-space filter than simple dilation. citeturn12view4turn42view0 | **directly implementable now** | citeturn12view4turn42view0 |
| 4 | **3D Gaussian Splatting for Real-Time Rendering of Radiance Fields** | SIGGRAPH 2023 | Your current baseline. Still the best reference for explicit Gaussian optimization, anisotropic covariance, density control, and visibility-aware rasterization. citeturn21search17 | **directly implementable now** | citeturn21search17turn0search2 |
| 5 | **Surface Splatting** | SIGGRAPH 2001 | Classic surfel paper that still matters because it gives you EWA-style anti-aliasing intuition and the core “surface element” worldview that 2DGS modernizes. citeturn0search8turn0search11 | **directly implementable now** | citeturn0search8turn0search11 |
| 6 | **StopThePop** | arXiv 2024 plus official code | Best rendering-side paper for view-consistent sorting and culling. If your 9-view optimization suffers view-dependent popping or unstable ordering, this is one of the most relevant rasterization upgrades. citeturn12view3 | **directly implementable now** | citeturn12view3turn4search8 |
| 7 | **Understanding Pure CLIP Guidance for Voxel Grid NeRF Models** | ICLR 2023 workshop / OpenReview | The best direct study of augmentation and anti-adversarial mechanics under pure CLIP guidance. It supports background augmentation, DiffAug-style image transforms, and model/representation choices that regularize geometry. citeturn22search0turn22search2turn22search6 | **directly implementable now** | citeturn22search0turn22search2turn22search6 |
| 8 | **High-quality Surface Reconstruction using Gaussian Surfels** | SIGGRAPH 2024 | Very relevant surfel variant: flatten the third scale to zero, use normal-depth consistency, and explicitly bias optimization toward surfaces. Especially useful if you adopt a hybrid 3DGS→surfel schedule. citeturn20search0turn20search3 | **useful later** | citeturn20search0turn20search1 |
| 9 | **Gaussian Billboards: Expressive 2D Gaussian Splatting with Textures** | arXiv 2024 / Disney Research page 2025 | Strong argument for adding small per-splat textures to 2D splats once geometry is stable. This could reduce primitive count and improve high-frequency appearance, but it also increases opportunities for CLIP cheating if added too early. citeturn38search3turn38search2 | **useful later** | citeturn38search3turn38search2 |
| 10 | **BillBoard Splatting** | ICCV 2025 | A more ambitious textured-planar-primitive replacement for Gaussians. Very relevant if you later want a hybrid that keeps surface bias while raising appearance capacity. citeturn18view0turn12view2 | **useful later** | citeturn18view0turn38search11 |
| 11 | **SuGaR** | CVPR 2024 | Valuable mainly for its surface-alignment regularization mindset. Even if you never extract meshes in-browser, the idea of forcing Gaussians to organize around a surface is directly relevant. citeturn12view1 | **useful later** | citeturn12view1turn7search0 |
| 12 | **Gaussian Opacity Fields** | SIGGRAPH Asia 2024 | Useful if you later want stronger surface extraction or more principled geometry regularization from Gaussian opacity structure. More reconstruction-oriented than your current needs, but conceptually aligned. citeturn7search4turn20search12 | **useful later** | citeturn7search4turn7search1 |
| 13 | **GStex** | WACV 2025 | One of the cleaner 2DGS texture papers. The key value is decoupling geometry from appearance so a single primitive can carry more visual detail. citeturn28search1turn28search17 | **useful later** | citeturn28search1turn28search17 |
| 14 | **GaussianPro** | arXiv 2024 / MLR 2024 | Relevant for progressive densification and optimization stability. It is reconstruction-driven rather than CLIP-driven, but its progressive propagation idea is one of the better explicit optimization schedules in 3DGS literature. citeturn21search1turn21search4 | **useful later** | citeturn21search1turn21search4 |
| 15 | **3D Gaussian Splatting as Markov Chain Monte Carlo** | NeurIPS 2024 | Interesting replacement for clone/split densification. Potentially useful if your current densification produces unstable clouds or mode collapse, but not a first implementation target for WebGPU. citeturn19search16turn19search8 | **useful later** | citeturn19search16turn19search0 |
| 16 | **CLIPDraw** | NeurIPS 2022 | Best 2D CLIP optimization reference for your 2D branch. It shows how primitive-constrained differentiable graphics plus CLIP bias outputs toward simpler, more legible shapes. citeturn27search5turn27search0 | **directly implementable now** | citeturn27search5turn27search0 |
| 17 | **CLIP-Mesh** | SIGGRAPH Asia 2022 | Relevant as a text-guided explicit-surface baseline. Good for mesh-side regularization ideas and understanding where direct CLIP shape optimization can work or fail. citeturn26search1turn26search13 | **useful later** | citeturn26search1turn26search13 |
| 18 | **DreamFusion** | 2022 paper / widely used reference | Not browser-practical, but highly relevant for ablations: view-dependent prompting, opacity regularization, orientation loss, textureless shading, and the argument that geometry quality needs more than plain semantic supervision. citeturn25view0turn36view0 | **useful later** | citeturn24view0turn25view0turn36view0 |
| 19 | **Score Jacobian Chaining** | CVPR 2023 | A primary SDS alternative worth knowing, mainly to understand how multi-view gradients from a 2D generative prior aggregate into 3D optimization. Not remotely in-scope for a first WebGPU implementation, but conceptually important. citeturn29search8turn29search4 | **useful later** | citeturn29search8turn29search0turn29search4 |
| 20 | **DreamView** | ECCV 2024 | Best paper on balancing overall and view-specific text guidance. You should not copy its full training setup, but its framing is directly helpful for deciding how much per-view prompt customization to allow. citeturn31search8turn31search3 | **useful later** | citeturn31search8turn31search0turn31search3 |

## Answers to your specific design questions

### Should this project move from volumetric 3D Gaussians to 2DGS surfels

Not all at once. The strongest recommendation is **no immediate hard switch**, but **yes to a staged move toward surface-biased splats**. 2DGS is compelling because it replaces volumetric ellipsoids with oriented planar disks, gives view-consistent geometry, and adds depth-distortion and normal-consistency regularization that directly target the kinds of multi-view inconsistency and floaters you want to reduce. Gaussian Surfels makes the same case in even more explicit “flatten to surface” terms. citeturn17view0turn20search0

The reason not to hard-switch immediately is that the papers demonstrating 2DGS and Gaussian Surfels are trained from **multi-view RGB supervision**, not from CLIP-only semantic supervision. In your setting, the early phase of optimization still needs an easy way to “occupy space” and form a coarse object before surface normals and depth regularizers have something meaningful to latch onto. Dream Fields specifically found that insufficient viewpoint diversity creates flat billboard-like solutions, which is a warning that a strongly surface-only parameterization can become brittle from scratch under weak supervision. citeturn23view0

A practical compromise is: **coarse 3DGS first, surfelization second**. Concretely, keep your volumetric Gaussians during the earliest phase, let opacity and pose regularizers establish a centered object, then progressively flatten covariances on stable splats, introduce normal consistency, and eventually render them like 2DGS surfels. That is not a paper-prescribed schedule, but it is the most defensible synthesis of the evidence for your constraints. citeturn21search17turn17view0turn20search0turn23view0

### Should you use 3DGS, 2DGS, billboards, or a hybrid

Use a **hybrid**.

Early training should stay close to **3DGS** because explicit 3D Gaussians are forgiving, visibility-aware, and already fit your current renderer. Mid training should transition toward **2DGS or surfel-like flattened splats** because that is where the best geometry and cross-view coherence gains come from. Only after geometry becomes stable should you experiment with **billboards or small per-splat textures**, because billboard-style primitives greatly increase per-primitive appearance capacity and therefore also increase the model’s ability to satisfy CLIP semantically without earning the underlying geometry. The billboard papers themselves motivate textured planar primitives as a way to raise expressivity and reduce primitive count, but that is exactly why they are better as a later-stage upgrade than as an early-stage CLIP-only primitive. citeturn21search17turn17view0turn20search0turn38search3turn18view0

So the recommended stack is:

- **Warm start:** anisotropic 3D Gaussians.
- **Surface phase:** 2DGS-style disks or surfel-flattened Gaussians.
- **Appearance phase:** optional tiny textures per primitive, billboard style, only after silhouette and coarse geometry are already good. citeturn17view0turn20search0turn38search3turn18view0

### What regularizers should be added first

The first batch should be these five, in this order.

**Opacity and transmittance control.** Dream Fields’ strongest transferable result is that a transmittance target plus background augmentation sharpens objects and prevents cloudy, semi-transparent failure modes. DreamFusion independently uses accumulated-alpha regularization to discourage filling empty space. This is your highest-priority fix for transparent holes and background cheating. citeturn23view0turn36view0turn36view1

**Scene bounds and object centering.** Dream Fields bounded density to a cube and tracked object location to prevent drift. In your explicit setting, a soft radial bound or bounding box prior around the origin should be easy to implement and will stop CLIP from “painting the frame” instead of making an object. citeturn23view0

**Anti-aliasing and frequency control.** Mip-Splatting is immediately relevant because CLIP is easily exploited by degenerate high-frequency artifacts. Add a simple screen-space mip-style filter or at least an EWA-inspired footprint model, and constrain splat scales from shrinking below what your rendered sampling rate can represent. citeturn12view4turn42view0turn0search8turn0search11

**Surface alignment.** Once the object exists, introduce 2DGS-style normal consistency and depth distortion, or Gaussian-Surfels-style normal-depth consistency. This is the most direct route to better geometry under weak supervision. citeturn17view0turn42view1turn20search0

**View-facing normal regularization.** If and only if you move toward surfels or flattened anisotropic Gaussians, a DreamFusion-style orientation prior can help keep visible normals from turning away and producing pathological shading or silhouette behavior. citeturn36view0turn36view2

### Is random background useful, or should black stay fixed

Random background is useful, and fixed black should **not** be the full-training default. Dream Fields is unusually explicit about this: compositing against a simple fixed white or black background causes the optimization to populate the background, while random backgrounds combined with transmittance regularization lead to coherent objects. citeturn23view0

For your art direction, I would **not** copy Dream Fields literally with bright random Fourier textures for every step. Instead, use a **dark-background curriculum**:

- early phase: mostly black, charcoal, navy, very dark procedural noise, and blurred dark textures;
- middle phase: mix in a smaller amount of brighter random procedural backgrounds to prevent trivial black-background exploits;
- final phase: short black-only refinement.  

That schedule is an inference, but it follows directly from Dream Fields’ evidence that some background diversity is important, while also respecting your intended “object on black” output. citeturn23view0

### Is CLIP batching worth implementing

It is worth implementing **after** the regularizers above, and **only if profiling says CLIP dominates** wall-clock time.

Why not first: your current biggest quality bottlenecks are almost certainly not raw MobileCLIP latency, but bad optimization incentives. Dream Fields and PureCLIPNeRF both show that augmentation and regularization materially change what CLIP-guided systems converge to. citeturn23view0turn22search0

Why still yes later: MobileCLIP is explicitly optimized for runtime efficiency and the official implementation uses standard batched encoding APIs; separately, recent WebGPU measurements show that dispatch-heavy batch-1 ML workloads can be bottlenecked by per-operation overhead, and that reducing dispatch count or amortizing work can help. That dispatch paper is not a CLIP paper, so this is an informed engineering analogy rather than direct evidence for your exact pipeline. Still, it strongly suggests that **micro-batching multiple views into one encoder/backward pass is the right second-phase optimization** if your WGSL MobileCLIP path is dispatch-heavy. citeturn15view0turn15view1turn33view0

My practical recommendation is:

- first, keep text embeddings precomputed and fixed;
- second, if you need a speed win before true batching, **sample a subset of your 9 cameras each step** and evaluate all 9 only periodically;
- third, if CLIP remains dominant, implement micro-batches of 3 or 9 views.  

The camera-subset recommendation is an inference from Dream Fields’ evidence that randomized view sampling improves geometry. citeturn23view0

### Are view-specific prompts helpful, or should all views use the same prompt

Use the **same base prompt for all views**, with **optional light directional modifiers** on canonical azimuth bins.

DreamFusion found view-dependent prompting beneficial and reports that without it, geometry was worse and could become multi-faced; it specifically used front, side, and back prompt augmentations chosen by camera azimuth. DreamView’s later work also frames the challenge as balancing overall text with view-specific text, not replacing the overall description outright. citeturn25view0turn25view2turn31search8turn31search3

For your CLIP-only setup, that points to a conservative policy:

- default to one shared base prompt across all 9 views;
- if the concept has a canonical orientation, use a small discrete set of suffixes such as “front view,” “side view,” and “back view”;
- avoid nine fully custom natural-language prompts unless you want intentional asymmetry.  

This preserves semantic coherence while still giving CLIP an orientation cue. citeturn25view0turn25view2turn31search8

## Recommended roadmap

The next five engineering steps I would take are these.

1. **Add Dream Fields–style transmittance and scene-bound losses, plus a dark random-background curriculum.** This is the highest-leverage change for preventing black-background cheating, diffuse clouds, and frame-filling artifacts. citeturn23view0

2. **Replace simple screen-space dilation with an alias-aware footprint.** A Mip-Splatting-inspired 2D mip filter, or at minimum an EWA-like footprint, should make both your renderings and your CLIP gradients less exploitable by tiny degenerate splats. citeturn12view4turn42view0turn0search8

3. **Move from isotropic 3D Gaussians to anisotropic Gaussians, then introduce a flattening schedule toward surfels.** Once the object is coherent, start penalizing thickness on high-opacity splats and add normal consistency or normal-depth consistency. This should improve geometry without requiring a full pure-2DGS rewrite on day one. citeturn21search17turn17view0turn20search0

4. **Simplify the prompting strategy to one shared prompt plus coarse directional bins, and consider stochastic view subsets per step.** Dream Fields supports randomized view exposure; DreamFusion supports light azimuth-conditioned prompt suffixes. This should reduce CLIP calls and improve coherence more reliably than nine bespoke prompts. citeturn23view0turn25view0

5. **Only after the above, prototype textured surfels or billboards.** Start with very small textures per primitive and only on stable surface primitives. The billboard papers are promising, but they are appearance-capacity upgrades, not first-line geometry fixes. citeturn38search3turn18view0turn28search1

## WGSL and WebGPU translation notes

The papers with the best **idea-to-WGSL** translation path are **3DGS, 2DGS, Mip-Splatting, Surface Splatting, StopThePop, Dream Fields, and CLIPDraw**. Their core contributions are either explicit primitive math, explicit rasterization logic, or explicit loss terms rather than giant training pipelines. citeturn21search17turn17view0turn42view0turn0search8turn12view3turn23view0turn27search5

The most useful code references, even when they are CUDA-based, are the official **gaussian-splatting** implementation, **2DGS**, **Mip-Splatting**, **BBSplat**, **GStex**, and **gsplat**. You cannot drop them into a browser, but they are still valuable for rasterization equations, tile structures, parameter layouts, and training-time bookkeeping. The important caveat is that **gsplat explicitly describes itself as CUDA-accelerated with Python bindings**, so it is a **reference implementation**, not a deployment path. citeturn0search2turn20search2turn42view0turn38search5turn28search17turn19search14turn19search6

The papers that are **much less translatable** to your constraint set are the diffusion-based text-to-3D papers such as **DreamFusion, SJC, ProlificDreamer, Fantasia3D, and Stable-DreamFusion**, because they rely on pretrained diffusion U-Nets, PyTorch-heavy pipelines, and usually CUDA-first codebases. They are still worth mining for ablations and regularizers, but not for a first browser implementation. citeturn24view0turn29search8turn40search0turn29search1turn10search10

## Warning list

Several things that look relevant at first glance are easy to overvalue for this project.

- **CLIP pretraining papers are not optimizer papers.** OpenAI CLIP, MobileCLIP2, and general CLIP scaling papers tell you about encoder training and deployment tradeoffs, not how to make a CLIP-guided splat optimizer form better geometry. MobileCLIP itself is relevant because you are already using it as the runtime encoder, but papers about improving CLIP pretraining are not where your next geometry wins will come from. citeturn34search11turn15view0turn34search6

- **CLIP2Scene and CLIP goes 3D are about recognition and 3D understanding, not text-guided 3D generation.** They may sound aligned because they combine CLIP and 3D, but they are the wrong subfield for your immediate needs. citeturn34search4turn34search16turn34search1

- **LLM quantization papers are irrelevant here.** GPTQ, AWQ, and LLM.int8 are about compressing transformer language models for inference. They do not improve CLIP-guided splat rendering, CLIP loss quality, or Gaussian optimization. citeturn35search0turn35search5turn35search2

- **Many of the most famous text-to-3D repos are CUDA-first and server-scale.** Stable-DreamFusion explicitly describes itself as a PyTorch implementation powered by Stable Diffusion, and gsplat explicitly describes itself as CUDA-accelerated. Those are fine idea references, but not “directly implementable now” for a browser-first WGSL system. citeturn10search10turn19search6

- **Some seemingly relevant “2.5DGS” material is not a peer-reviewed primary paper.** The GitHub project commonly cited for “2.5DGS” is an unofficial implementation path based on existing Gaussian renderers, which can still be useful as an experiment but should not be treated as a fully validated literature result. citeturn21search2turn39view0

## Open questions and limitations

The biggest unresolved gap in the literature for your exact use case is that **there is no canonical paper for CLIP-only optimization of explicit 3D Gaussian splats in a browser runtime**. The strongest guidance therefore comes from triangulating between CLIP-only NeRF papers, explicit Gaussian reconstruction papers, and classic surfel/rasterization work instead of from a single perfect match. citeturn23view0turn22search0turn17view0turn21search17

A second limitation is that the strongest surfel papers, including 2DGS and Gaussian Surfels, were validated under **multi-view image reconstruction losses**, not under CLIP-only supervision. That is why the recommendation here is a **hybrid staged transition** rather than an immediate pure-2DGS rewrite. citeturn17view0turn20search0

A third limitation is that the evidence for **WebGPU batching gains** is indirect for your exact model. MobileCLIP is clearly runtime-oriented, and WebGPU dispatch overhead is clearly real, but I did not find a primary source benchmarking MobileCLIP-S0 forward-plus-backward under browser WebGPU multi-image batching specifically. So the batching recommendation should be treated as a profiling-driven engineering decision, not a guaranteed win. citeturn15view0turn15view1turn33view0