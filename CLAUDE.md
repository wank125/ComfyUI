# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI is a powerful and modular visual AI engine that provides a graph/nodes/flowchart-based interface for designing and executing advanced stable diffusion pipelines. It allows users to create complex workflows without writing code by connecting different functional nodes in a visual interface.

## Common Development Commands

### Running ComfyUI

```bash
# Basic run
python main.py

# With specific settings
python main.py --listen 0.0.0.0 --port 8188

# Enable verbose/debug logging
python main.py --verbose DEBUG

# Enable ComfyUI-Manager
python main.py --enable-manager

# CPU-only mode (for testing)
python main.py --cpu

# Development mode with auto-launch
python main.py --auto-launch --verbose
```

### Testing

```bash
# Install test dependencies
pip install -r tests-unit/requirements.txt

# Run unit tests
python -m pytest tests-unit

# Run integration tests (requires additional dependencies)
pip install pytest websocket-client==1.6.1 opencv-python==4.6.0.66 scikit-image==0.21.0
python -m pytest tests/inference

# Run all tests
python -m pytest
```

### Code Quality

```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Run pylint (on specific modules)
pylint comfy_api_nodes
```

### Installing Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# GPU-specific PyTorch (NVIDIA)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

# GPU-specific PyTorch (AMD ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# GPU-specific PyTorch (Intel Arc)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

## High-Level Architecture

### Core Components

1. **main.py**: Entry point that handles CLI arguments, environment setup, and server initialization
2. **server.py**: Async HTTP server using aiohttp, handles WebSocket connections for real-time updates
3. **execution.py**: Workflow execution engine with queue management and caching
4. **nodes.py**: Core node implementations (LoadCheckpoint, CLIPTextEncode, etc.)
5. **comfy/**: Core modules containing model implementations, samplers, and utilities

### Key Directories

- `comfy/`: Core engine modules
  - `ldm/`: Latent diffusion models (SD1.x, SDXL, SD3, Flux, etc.)
  - `k_diffusion/`: Sampling algorithms
  - `samplers/`: Sampling method implementations
  - `latent_formats.py`: Model format handling
- `app/`: Application layer (database, model management, user management)
- `api_server/`: REST API server implementation
- `tests/`: Integration tests
- `tests-unit/`: Unit tests
- `web/`: Frontend static files (now a separate repo, synced periodically)

### Model Support

ComfyUI supports multiple model architectures:
- **Image Models**: SD1.x, SD2.x, SDXL, SD3, Flux, Stable Cascade, AuraFlow, HunyuanDiT, etc.
- **Video Models**: SVD, Mochi, LTX-Video, Hunyuan Video, Wan
- **Audio Models**: Stable Audio, ACE Step
- **3D Models**: Hunyuan3D

### Workflow System

- Node-based graph interface stored as JSON
- Optimized execution (only changed parts re-run)
- Smart caching with multiple strategies (classic, LRU, RAM pressure)
- Async queue system for handling multiple requests

### Memory Management Options

- `--lowvram`: Split unet to use less VRAM
- `--highvram`: Keep models in GPU memory
- `--gpu-only`: Store everything on GPU
- `--cpu`: Use CPU for everything
- `--reserve-vram`: Reserve VRAM for OS

## Important Development Patterns

### Adding New Nodes

1. Create a new class inheriting from `comfy.node.Node`
2. Implement `INPUT_TYPES()` classmethod to define inputs
3. Implement the main execution method
4. Register the node using `NODE_CLASS_MAPPINGS`

### Model Loading

Models are loaded through the `comfy.model_management` system which handles:
- Device placement
- Memory offloading
- Model caching
- Format conversion (ckpt, safetensors)

### Custom Nodes

Custom nodes are loaded from the `custom_nodes` directory. Each custom node should:
- Have its own directory under `custom_nodes/`
- Include an `__init__.py` that registers nodes
- Handle dependencies properly

### Frontend Integration

The frontend is now in a separate repository (ComfyUI_frontend). To use the latest version:
```bash
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@latest
```

## Configuration

### Model Paths

Configure model search paths in `extra_model_paths.yaml` (copy from `extra_model_paths.yaml.example`).

### Environment Variables

- `HF_HUB_DISABLE_TELEMETRY=1`: Disable HuggingFace telemetry
- `DO_NOT_TRACK=1`: Disable tracking
- `HSA_OVERRIDE_GFX_VERSION`: For AMD GPU compatibility

## Development Tips

1. **Testing Changes**: Use the preview functionality (`--preview-method auto`) for quick iteration
2. **Debugging**: Enable verbose logging with `--verbose DEBUG`
3. **Memory Issues**: Use `--lowvram` or `--cpu` for debugging on systems with limited VRAM
4. **Hot Reloading**: The server automatically detects and loads new custom nodes
5. **API Usage**: ComfyUI provides a REST API and WebSocket for external integration

## Release Process

ComfyUI follows a weekly release cycle targeting Mondays. Three interconnected repositories:
1. **ComfyUI Core** (this repo) - stable releases weekly
2. **ComfyUI Desktop** - builds using latest stable core
3. **ComfyUI Frontend** - weekly updates merged into core

## GPU Support

- **NVIDIA**: CUDA support with various PyTorch versions
- **AMD**: ROCm support (Linux) and experimental Windows support
- **Intel**: Arc GPU support with oneAPI
- **Apple Silicon**: MPS support via PyTorch nightly
- **Ascend**: NPU support via torch_npu