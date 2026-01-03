# NeRF - Neural Radiance Fields

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

### Novel View Synthesis with Neural Radiance Fields

_Transform 2D images into continuous 3D scene representations_  
_using state-of-the-art volume rendering_

---

## Features

| Module              | Paper Reference | Function             |
| ------------------- | --------------- | -------------------- |
| Camera Model        | Section 4       | `get_rays()`         |
| Stratified Sampling | Equation (2)    | `sample_points()`    |
| Volume Rendering    | Equation (3)    | `volume_rendering()` |
| Raw Processing      | —               | `process_raw()`      |

## Architecture

```
NeRF MLP: F_Θ(x, d) → (σ, c)
├── 8 layers, 256 channels
├── Skip connection at layer 4
├── Density: ReLU (σ >= 0)
└── Color: Sigmoid (c ∈ [0,1])
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook nerf_forward_pass.ipynb
```

---

## Pipeline

```
Pixel (u,v) → get_rays() → sample_points() → NeRF MLP → process_raw() → volume_rendering()
                ↓               ↓                ↓            ↓              ↓
           rays_o, rays_d    pts, z_vals      raw[...,4]   rgb, sigma    rgb, depth, acc
```

---

## Outputs

| Output | Shape       | Description              |
| ------ | ----------- | ------------------------ |
| RGB    | `[H, W, 3]` | Rendered color image     |
| Depth  | `[H, W]`    | Expected ray termination |
| Acc    | `[H, W]`    | Accumulated opacity      |

---

## 📁 Project Structure

```
NeRF/
├── nerf_forward_pass.ipynb    # Main implementation
├── DEEP_THEORY_GUIDE.md       # Theory documentation
├── SIMPLIFIED_TASK_BREAKDOWN.md
├── PROJECT_ROADMAP.md
├── CODE_EXPLANATION.md
├── LEARNING_GUIDE.md
├── 2003.08934v2.pdf           # Original paper
└── requirements.txt
```

---

## Key Functions

### `get_rays(H, W, K, c2w)`

Generates rays for all pixels using pinhole camera model.

### `sample_points(rays_o, rays_d, near, far, N_samples)`

Stratified sampling along rays — Equation (2).

### `process_raw(raw)`

Applies Sigmoid to RGB, ReLU to density.

### `volume_rendering(rgb, sigma, z_vals, rays_d)`

Volume rendering — Equation (3). Returns `rgb_map`, `depth_map`, `acc_map`.

### `render_image(H, W, K, c2w, model, near, far, N_samples)`

Complete rendering pipeline.

---

## 🛠️ Tech Stack

- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Jupyter Notebook** - Interactive development

---

## References

- [NeRF Paper (arXiv)](https://arxiv.org/abs/2003.08934)
- [NeRF Project Page](https://www.matthewtancik.com/nerf)
- [Original Implementation](https://github.com/bmild/nerf)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## License

This project is for educational purposes.

Built as part of **GDSC AIML learning track**
