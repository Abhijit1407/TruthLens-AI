\# 🛠️ Installation Guide - TruthLens AI

Complete step-by-step installation instructions for TruthLens AI multimodal fake news detection system.

\---

\## 📋 Table of Contents

1\. \[System Requirements\](#system-requirements)

2\. \[Installation Methods\](#installation-methods)

3\. \[Model Download\](#model-download)

4\. \[Verification\](#verification)

5\. \[Troubleshooting\](#troubleshooting)

6\. \[GPU Setup\](#gpu-setup)

\---

\## 💻 System Requirements

\### Minimum Requirements

\- \*\*OS:\*\* Windows 10+, macOS 10.14+, Ubuntu 18.04+

\- \*\*Python:\*\* 3.8 or higher

\- \*\*RAM:\*\* 8GB minimum

\- \*\*Storage:\*\* 2GB free disk space

\- \*\*Internet:\*\* For initial setup and model downloads

\### Recommended Requirements

\- \*\*RAM:\*\* 16GB or higher

\- \*\*GPU:\*\* NVIDIA GPU with CUDA support (Tesla T4, RTX 3060+, or better)

\- \*\*VRAM:\*\* 4GB+ for optimal performance

\- \*\*Storage:\*\* 5GB+ for datasets and models

\### Check Python Version

\`\`\`bash

python --version

\# Should show: Python 3.8.x or higher

\# If not installed, download from:

\# https://www.python.org/downloads/

\`\`\`

\---

\## 📦 Installation Methods

\### Method 1: Quick Install (Recommended)

\*\*Step 1: Clone Repository\*\*

\`\`\`bash

\# Using HTTPS

git clone https://github.com/YOUR\_USERNAME/TruthLens-AI.git

cd TruthLens-AI

\# OR using SSH

git clone git@github.com:YOUR\_USERNAME/TruthLens-AI.git

cd TruthLens-AI

\`\`\`

\*\*Step 2: Create Virtual Environment (Recommended)\*\*

\`\`\`bash

\# Create virtual environment

python -m venv venv

\# Activate it

\# On Windows:

venv\\Scripts\\activate

\# On macOS/Linux:

source venv/bin/activate

\# You should see (venv) in your terminal prompt

\`\`\`

\*\*Step 3: Install Dependencies\*\*

\`\`\`bash

\# Install all required packages

pip install -r requirements.txt

\# This installs:

\# - PyTorch and TorchVision

\# - Transformers (HuggingFace)

\# - Gradio

\# - NumPy, Pandas, Matplotlib, Seaborn

\# - scikit-learn

\# - Pillow

\`\`\`

\*\*Step 4: Download Models\*\* (see next section)

\---

\### Method 2: Manual Installation

If you prefer to install packages individually:

\`\`\`bash

\# Core ML frameworks

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install transformers

\# UI and utilities

pip install gradio

pip install pillow numpy pandas scikit-learn

\# Visualization

pip install matplotlib seaborn

\# Optional but recommended

pip install jupyter notebook ipywidgets

\`\`\`

\---

\### Method 3: Conda Installation

For Anaconda/Miniconda users:

\`\`\`bash

\# Create conda environment

conda create -n truthlens python=3.10

conda activate truthlens

\# Install PyTorch with CUDA

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

\# Install other packages

pip install transformers gradio

pip install scikit-learn pandas matplotlib seaborn pillow

\# Verify installation

python -c "import torch; print(f'PyTorch: {torch.\_\_version\_\_}')"

python -c "import transformers; print(f'Transformers: {transformers.\_\_version\_\_}')"

\`\`\`

\---

\## 📥 Model Download

\### Large Model Files

Due to GitHub's file size limitations, some models are hosted externally:

\#### Text Models

\*\*BERT Model (Required for text analysis):\*\*

\- \*\*File:\*\* \`bert\_best\_model.pt\`

\- \*\*Size:\*\* 417.7 MB

\- \*\*Download:\*\* \[Google Drive Link\](YOUR\_GOOGLE\_DRIVE\_LINK)

\- \*\*Destination:\*\* \`models/text\_models/bert\_best\_model.pt\`

\*\*Random Forest Baseline (Optional):\*\*

\- \*\*File:\*\* \`random\_forest\_model.pkl\`

\- \*\*Size:\*\* 232.6 MB

\- \*\*Download:\*\* \[Google Drive Link\](YOUR\_GOOGLE\_DRIVE\_LINK)

\- \*\*Destination:\*\* \`models/text\_models/random\_forest\_model.pkl\`

\#### Image Models

\*\*EfficientNet Model (Included in repo):\*\*

\- \*\*File:\*\* \`best\_efficientnet\_cifake\_fast.pth\`

\- \*\*Size:\*\* 46.4 MB

\- \*\*Location:\*\* Already in \`models/image\_models/\`

\- \*\*No download needed\*\* ✅

\### Download Script

\*\*Quick download helper:\*\*

\`\`\`bash

\# Create downloads folder

mkdir -p downloads

\# Download BERT (replace with your actual link)

\# Using wget:

wget -O downloads/bert\_best\_model.pt "YOUR\_GOOGLE\_DRIVE\_LINK"

\# OR using curl:

curl -L -o downloads/bert\_best\_model.pt "YOUR\_GOOGLE\_DRIVE\_LINK"

\# Move to correct location

mv downloads/bert\_best\_model.pt models/text\_models/

\`\`\`

\### Manual Download Steps

1\. Click the Google Drive link above

2\. Click "Download" button

3\. Save file to \`TruthLens-AI/models/text\_models/\`

4\. Verify file size matches expected size

5\. Rename if necessary to match exact filename

\---

\## ✅ Verification

\### Verify Installation

\*\*Check Python Packages:\*\*

\`\`\`bash

python -c "

import torch

import transformers

import gradio

import PIL

import sklearn

import matplotlib

print('✓ PyTorch:', torch.\_\_version\_\_)

print('✓ Transformers:', transformers.\_\_version\_\_)

print('✓ Gradio:', gradio.\_\_version\_\_)

print('✓ All packages installed successfully!')

"

\`\`\`

\*\*Expected Output:\*\*

\`\`\`

✓ PyTorch: 2.x.x

✓ Transformers: 4.x.x

✓ Gradio: 4.x.x

✓ All packages installed successfully!

\`\`\`

\### Verify Models

\*\*Check Model Files:\*\*

\`\`\`bash

\# On Unix/Mac:

ls -lh models/text\_models/bert\_best\_model.pt

ls -lh models/image\_models/best\_efficientnet\_cifake\_fast.pth

\# On Windows:

dir models\\text\_models\\bert\_best\_model.pt

dir models\\image\_models\\best\_efficientnet\_cifake\_fast.pth

\`\`\`

\*\*Expected:\*\*

\- BERT: ~417 MB

\- EfficientNet: ~46 MB

\### Test Models

\*\*Quick model test:\*\*

\`\`\`python

import torch

from transformers import BertTokenizer, BertForSequenceClassification

\# Test BERT loading

device = 'cuda' if torch.cuda.is\_available() else 'cpu'

print(f"Using device: {device}")

tokenizer = BertTokenizer.from\_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from\_pretrained('bert-base-uncased', num\_labels=2)

model.load\_state\_dict(torch.load('models/text\_models/bert\_best\_model.pt', map\_location=device))

print("✓ BERT model loaded successfully!")

\# Test prediction

text = "Test headline"

encoding = tokenizer(text, return\_tensors='pt', padding=True, truncation=True, max\_length=128)

with torch.no\_grad():

outputs = model(\*\*encoding)

probs = torch.softmax(outputs.logits, dim=1)

print(f"✓ Inference working! Probabilities: {probs.numpy()}")

\`\`\`

\*\*If this runs without errors, you're ready to go!\*\* ✅

\---

\## 🔧 Troubleshooting

\### Common Issues

\#### Issue 1: "No module named 'torch'"

\*\*Solution:\*\*

\`\`\`bash

\# Reinstall PyTorch

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

\`\`\`

\#### Issue 2: "CUDA not available" or "GPU not detected"

\*\*Solution:\*\*

\`\`\`bash

\# Check CUDA availability

python -c "import torch; print('CUDA available:', torch.cuda.is\_available())"

\# If False, either:

\# Option A: Install CUDA-enabled PyTorch

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

\# Option B: Use CPU version (slower)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

\`\`\`

\#### Issue 3: "FileNotFoundError: bert\_best\_model.pt"

\*\*Solution:\*\*

\`\`\`bash

\# Check file exists

ls models/text\_models/bert\_best\_model.pt

\# If not found:

\# 1. Re-download from Google Drive

\# 2. Verify you placed it in correct folder: models/text\_models/

\# 3. Check filename matches exactly (case-sensitive)

\`\`\`

\#### Issue 4: "RuntimeError: CUDA out of memory"

\*\*Solution:\*\*

\`\`\`python

\# In your code, reduce batch size or use CPU

device = 'cpu' # Force CPU usage

\# OR add this to free GPU memory

import torch

torch.cuda.empty\_cache()

\`\`\`

\#### Issue 5: "transformers not found"

\*\*Solution:\*\*

\`\`\`bash

pip install transformers

\# OR

pip install transformers\[torch\]

\`\`\`

\#### Issue 6: Gradio port already in use

\*\*Solution:\*\*

\`\`\`bash

\# In chatbot.py, change port:

demo.launch(server\_port=7861) # Try different port

\`\`\`

\### Platform-Specific Issues

\#### Windows

\*\*Issue:\*\* "pip is not recognized"

\`\`\`bash

\# Solution: Use python -m pip

python -m pip install -r requirements.txt

\`\`\`

\*\*Issue:\*\* Long path errors

\`\`\`bash

\# Enable long paths in Windows

\# Run PowerShell as Administrator:

New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

\`\`\`

\#### macOS

\*\*Issue:\*\* "xcrun: error: invalid active developer path"

\`\`\`bash

\# Install Xcode Command Line Tools

xcode-select --install

\`\`\`

\#### Linux

\*\*Issue:\*\* Missing system libraries

\`\`\`bash

\# Install required system packages

sudo apt-get update

sudo apt-get install python3-pip python3-dev build-essential

\`\`\`

\---

\## 🎮 GPU Setup (Optional but Recommended)

\### Check GPU Availability

\`\`\`python

import torch

print("CUDA Available:", torch.cuda.is\_available())

print("CUDA Version:", torch.version.cuda)

if torch.cuda.is\_available():

print("GPU Name:", torch.cuda.get\_device\_name(0))

print("GPU Memory:", torch.cuda.get\_device\_properties(0).total\_memory / 1e9, "GB")

else:

print("Running on CPU (slower but works)")

\`\`\`

\### Install CUDA Toolkit (If Needed)

\*\*NVIDIA GPU Users:\*\*

1\. \*\*Check your GPU:\*\* \[https://www.nvidia.com/Download/index.aspx\](https://www.nvidia.com/Download/index.aspx)

2\. \*\*Download CUDA Toolkit:\*\* \[https://developer.nvidia.com/cuda-downloads\](https://developer.nvidia.com/cuda-downloads)

3\. \*\*Install PyTorch with CUDA:\*\*

\`\`\`bash

\# For CUDA 11.8

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

\# For CUDA 12.1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

\`\`\`

\### Performance Comparison

| Device | Text Analysis Time | Image Analysis Time |

|--------|-------------------|---------------------|

| \*\*CPU (Intel i7)\*\* | ~2-3 seconds | ~3-5 seconds |

| \*\*GPU (Tesla T4)\*\* | ~0.5 seconds | ~0.3 seconds |

| \*\*GPU (RTX 3060)\*\* | ~0.3 seconds | ~0.2 seconds |

\*\*Note:\*\* System works on CPU but GPU is 5-10× faster.

\---

\## 🐳 Docker Installation (Advanced)

For containerized deployment:

\*\*Dockerfile (create in root):\*\*

\`\`\`dockerfile

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD \["python", "src/chatbot.py"\]

\`\`\`

\*\*Build and Run:\*\*

\`\`\`bash

\# Build image

docker build -t truthlens-ai .

\# Run container

docker run -p 7860:7860 truthlens-ai

\# Access at http://localhost:7860

\`\`\`

\---

\## 📊 Disk Space Requirements

| Component | Size | Required? |

|-----------|------|-----------|

| \*\*Python packages\*\* | ~2.5 GB | ✅ Yes |

| \*\*BERT model\*\* | 417.7 MB | ✅ Yes (text analysis) |

| \*\*EfficientNet model\*\* | 46.4 MB | ✅ Yes (image analysis) |

| \*\*Random Forest\*\* | 232.6 MB | ⚪ Optional (baseline) |

| \*\*Other baselines\*\* | <1 MB each | ⚪ Optional |

| \*\*Notebooks & results\*\* | ~50 MB | ⚪ Optional |

| \*\*Datasets (if downloaded)\*\* | ~2 GB | ⚪ Optional |

| \*\*Total Minimum\*\* | ~3 GB | - |

| \*\*Total Full Install\*\* | ~5-6 GB | - |

\---

\## 🔐 API Keys & Credentials

\### Required

\- \*\*None!\*\* TruthLens AI runs completely offline after installation.

\### Optional (for development)

\- \*\*Kaggle API:\*\* For downloading CIFAKE dataset

\`\`\`bash

\# Place kaggle.json in ~/.kaggle/

mkdir -p ~/.kaggle

cp /path/to/kaggle.json ~/.kaggle/

chmod 600 ~/.kaggle/kaggle.json

\`\`\`

\- \*\*HuggingFace Token:\*\* For accessing some models

\`\`\`bash

\# Login to HuggingFace (optional)

huggingface-cli login

\`\`\`

\---

\## 📦 Dependencies Explained

\### Core Dependencies

| Package | Purpose | Size |

|---------|---------|------|

| \*\*torch\*\* | Deep learning framework | ~700 MB |

| \*\*torchvision\*\* | Image transformations | ~300 MB |

| \*\*transformers\*\* | BERT and tokenizers | ~500 MB |

| \*\*gradio\*\* | Web interface | ~100 MB |

\### Supporting Libraries

| Package | Purpose |

|---------|---------|

| \*\*Pillow\*\* | Image loading and processing |

| \*\*numpy\*\* | Numerical computations |

| \*\*pandas\*\* | Data manipulation |

| \*\*scikit-learn\*\* | ML utilities and metrics |

| \*\*matplotlib\*\* | Plotting and visualization |

| \*\*seaborn\*\* | Statistical visualizations |

\---

\## ✅ Post-Installation Checklist

After installation, verify everything works:

\- \[ \] Python version is 3.8+

\- \[ \] Virtual environment activated

\- \[ \] All packages installed (\`pip list\`)

\- \[ \] BERT model downloaded and placed correctly

\- \[ \] EfficientNet model exists in repo

\- \[ \] Test script runs without errors

\- \[ \] Chatbot launches successfully

\- \[ \] GPU detected (if available)

\*\*Run this verification script:\*\*

\`\`\`bash

python -c "

import sys

import torch

import transformers

import gradio

import os

print('Python version:', sys.version.split()\[0\])

print('PyTorch:', torch.\_\_version\_\_)

print('Transformers:', transformers.\_\_version\_\_)

print('Gradio:', gradio.\_\_version\_\_)

print('CUDA available:', torch.cuda.is\_available())

print('BERT model exists:', os.path.exists('models/text\_models/bert\_best\_model.pt'))

print('EfficientNet exists:', os.path.exists('models/image\_models/best\_efficientnet\_cifake\_fast.pth'))

if all(\[

torch.cuda.is\_available() or True, # CPU is OK

os.path.exists('models/text\_models/bert\_best\_model.pt'),

os.path.exists('models/image\_models/best\_efficientnet\_cifake\_fast.pth')

\]):

print('\\n✅ Installation verified! Ready to run.')

else:

print('\\n⚠️ Some components missing. Check above.')

"

\`\`\`

\---

\## 🌐 Google Colab Setup (Alternative)

Don't want to install locally? Run in Google Colab:

\*\*Step 1: Upload to Google Drive\*\*

\`\`\`python

from google.colab import drive

drive.mount('/content/drive')

\# Clone repo to Colab

!git clone https://github.com/YOUR\_USERNAME/TruthLens-AI.git

%cd TruthLens-AI

\`\`\`

\*\*Step 2: Install packages\*\*

\`\`\`python

!pip install -q -r requirements.txt

\`\`\`

\*\*Step 3: Copy models from Drive\*\*

\`\`\`python

\# If you have models in Drive

!cp /content/drive/MyDrive/YOUR\_PATH/bert\_best\_model.pt models/text\_models/

\`\`\`

\*\*Step 4: Run chatbot\*\*

\`\`\`python

!python src/chatbot.py

\`\`\`

\*\*Advantages:\*\*

\- ✅ Free GPU (Tesla T4)

\- ✅ No local installation needed

\- ✅ 15GB RAM available

\- ✅ Easy sharing with collaborators

\---

\## 🔄 Updating

\### Update Code

\`\`\`bash

\# Pull latest changes

git pull origin main

\# Update dependencies

pip install -r requirements.txt --upgrade

\`\`\`

\### Update Models

\`\`\`bash

\# Re-download models if updated

\# Check repository releases for new model versions

\`\`\`

\---

\## ❌ Uninstallation

\### Remove Virtual Environment

\`\`\`bash

\# Deactivate environment

deactivate

\# Remove directory

rm -rf venv/ # On Unix/Mac

rmdir /s venv # On Windows

\`\`\`

\### Remove Everything

\`\`\`bash

\# Navigate to parent directory

cd ..

\# Remove entire repository

rm -rf TruthLens-AI/ # On Unix/Mac

rmdir /s TruthLens-AI # On Windows

\`\`\`

\---

\## 🆘 Getting Help

\### Check Logs

\`\`\`bash

\# Enable verbose logging

python src/chatbot.py --verbose

\# Check Python logs

python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

\`\`\`

\### Report Issues

If you encounter problems:

1\. \*\*Check existing issues:\*\* \[GitHub Issues\](https://github.com/YOUR\_USERNAME/TruthLens-AI/issues)

2\. \*\*Create new issue:\*\* Provide:

\- OS and Python version

\- Error message (full traceback)

\- Steps to reproduce

\- What you've tried

\### Community Support

\- 💬 \[GitHub Discussions\](https://github.com/YOUR\_USERNAME/TruthLens-AI/discussions)

\- 📧 Email: your.email@northeastern.edu

\---

\## 🚀 Quick Setup Summary

\*\*For the impatient:\*\*

\`\`\`bash

\# 1. Clone

git clone https://github.com/YOUR\_USERNAME/TruthLens-AI.git && cd TruthLens-AI

\# 2. Install

pip install -r requirements.txt

\# 3. Download BERT model

\# (Use Google Drive link from repository)

\# 4. Run

python src/chatbot.py

\# Done! 🎉

\`\`\`

\---

\## 💡 Tips for Beginners

1\. \*\*Use virtual environments\*\* - Keeps dependencies isolated

2\. \*\*Start with CPU\*\* - Verify everything works before GPU setup

3\. \*\*Read error messages\*\* - They usually tell you exactly what's wrong

4\. \*\*Check file paths\*\* - Most errors are incorrect paths

5\. \*\*Ask for help\*\* - GitHub issues and discussions are there for you

\---

\## 🎓 For Instructors/Reviewers

\*\*Quick evaluation setup:\*\*

\`\`\`bash

\# 1. Clone repository

git clone https://github.com/YOUR\_USERNAME/TruthLens-AI.git

cd TruthLens-AI

\# 2. Install (5 minutes)

pip install -r requirements.txt

\# 3. Download pre-trained models

\# Links provided in repository README

\# 4. Launch demo

python src/chatbot.py

\# 5. Open browser to Gradio URL

\# Test with sample headlines!

\`\`\`

\*\*Expected demo time:\*\* 2-3 minutes from clone to working chatbot.

\---

\## 📞 Support

\*\*Having trouble?\*\*

1\. ✅ Check \[Troubleshooting\](#troubleshooting) section above

2\. ✅ Review \[USAGE.md\](USAGE.md) for common questions

3\. ✅ Search \[GitHub Issues\](https://github.com/YOUR\_USERNAME/TruthLens-AI/issues)

4\. ✅ Create new issue with details

\*\*For academic inquiries:\*\*

\- Team: Abhijit More, Kshama Upadhyay, Qiwei Guo

\- Course: EAI6010 Applications of AI

\- Institution: Northeastern University

\---

\*Last Updated: January 2026\*

\*Setup Guide Version: 1.0\*