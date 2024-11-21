# Local LLM Runner (GPU version)

**Version 1.0**
### Creator: Juhani Merilehto - @juhanimerilehto - Jyväskylä University of Applied Sciences (JAMK), Likes institute

![JAMK Likes Logo](./assets/likes_str_logo.png)

## Overview

Local LLM Runner is a Python-based tool that enables local execution of Large Language Models (LLMs) with GPU acceleration. It was developed for the Strategic Exercise Information and Research unit in Likes Institute, at JAMK University of Applied Sciences. The tool provides a simple command-line interface for interacting with various GGUF-format language models, with specific optimization for the OpenHermes 2.5 model.

## Features

- **Local Processing**: Fully local solution, no data sent to external servers
- **GPU Acceleration**: Automatic CUDA support for faster processing
- **Model Flexibility**: Easy swapping between different GGUF models
- **Interactive CLI**: Simple command-line interface for model interaction
- **Configurable Parameters**: Adjustable temperature, GPU layers, and context length

## Hardware Requirements

- **CPU:** Modern CPU for basic operation
- **GPU:** NVIDIA GPU with CUDA support recommended
  - Minimum 8GB VRAM for optimal performance
  - Tested with RTX 4060
- **RAM:** 32GB recommended

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/juhanimerilehto/local-llm-runner.git
cd local-llm-runner
```

### 2. Create a virtual environment:
```bash
python -m venv llm-env
source llm-env/bin/activate  # For Windows: llm-env\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install ctransformers huggingface_hub
```

## Usage

Run the script with default settings (OpenHermes 2.5):
```bash
python llm_runner.py
```

Or specify a different model:
```bash
python llm_runner.py --model_url "TheBloke/Mixtral-8x7B-v0.1-GGUF" --model_file "mixtral-8x7b-v0.1.Q4_K_M.gguf"
```

## File Structure

```plaintext
local-llm-runner/
├── assets/
│   └── jamklikes.png
├── llm_runner.py
├── requirements.txt
└── README.md
```

## Credits

- **Juhani Merilehto (@juhanimerilehto)** – Specialist, Data and Statistics
- **JAMK Likes** – Organization sponsor

## License

This project is licensed for free use under the condition that proper credit is given to Juhani Merilehto (@juhanimerilehto) and JAMK Likes institute. You are free to use, modify, and distribute this project, provided that you mention the original author and institution and do not hold them liable for any consequences arising from the use of the software.