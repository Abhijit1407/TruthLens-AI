\# 📖 Usage Guide - TruthLens AI

Complete guide for using TruthLens AI for fake news detection.

\---

\## 📋 Table of Contents

1\. \[Quick Start\](#quick-start)

2\. \[Using the Chatbot\](#using-the-chatbot)

3\. \[Programmatic Usage\](#programmatic-usage)

4\. \[Jupyter Notebooks\](#jupyter-notebooks)

5\. \[Advanced Usage\](#advanced-usage)

6\. \[API Reference\](#api-reference)

7\. \[Examples\](#examples)

\---

\## 🚀 Quick Start

\### Launch the Chatbot

\`\`\`bash

\# Navigate to project directory

cd TruthLens-AI

\# Activate virtual environment (if using one)

source venv/bin/activate # macOS/Linux

venv\\Scripts\\activate # Windows

\# Launch chatbot

python src/chatbot.py

\`\`\`

\*\*Expected Output:\*\*

\`\`\`

\======================================================================

🚀 TRUTHLENS AI - MULTIMODAL CHATBOT

\======================================================================

✓ Using device: cuda

📝 Loading BERT text model...

✅ BERT model loaded (84.5% accuracy)

🖼️ Loading EfficientNet image model...

✅ EfficientNet loaded (97.66% accuracy)

\======================================================================

🚀 LAUNCHING TRUTHLENS AI - COMPLETE SYSTEM

\======================================================================

Running on local URL: http://127.0.0.1:7860

Running on public URL: https://xxxxx.gradio.live

🎉 TruthLens AI is LIVE!

\`\`\`

\*\*Open the URL in your browser\*\* and start analyzing content!

\---

\## 💬 Using the Chatbot

\### Interface Overview

The TruthLens AI chatbot provides an intuitive conversational interface:

\`\`\`

┌─────────────────────────────────────────────────────────────┐

│ 🔍 TruthLens AI - Multimodal Fake News Detection │

│ Performance: 84.5% (Text) | 97.66% (Image) │

├─────────────────────────────────────────────────────────────┤

│ │

│ \[Chat conversation area\] │

│ │

├─────────────────────────────────────────────────────────────┤

│ \[Text input box\] \[📎 Upload Image\] │

│ \[🔍 Analyze Button\] │

├─────────────────────────────────────────────────────────────┤

│ Sample Buttons: 🚨 Fake ✅ Real 🚨 Fake ✅ Real │

│ Commands: ❓ Help ℹ️ About 📊 Stats │

└─────────────────────────────────────────────────────────────┘

\`\`\`

\### Basic Usage

\#### 1. Analyze Text Headlines

\*\*Type or paste any news headline:\*\*

\`\`\`

User: "BREAKING: Scientists discover miracle cure that works 100%!!!"

TruthLens AI:

🔍 Analysis Complete

🚨 HIGH RISK - Likely Fake News

This looks very suspicious to me.

Confidence: 92.3%

📝 Text Analysis

\- Real News: 7.7%

\- Fake News: 92.3%

Red Flags Detected:

\- Sensational: breaking

\- Exaggerated: miracle, 100%

\- Emotional: !!!

💡 Recommendation

⚠️ High confidence detection of misinformation.

I strongly suggest fact-checking through Snopes, PolitiFact, or Reuters.

\`\`\`

\#### 2. Upload Images (If Image Model Available)

\*\*Drag and drop or click upload:\*\*

\`\`\`

User: \[Uploads AI-generated robot image\]

TruthLens AI:

🔍 Analysis Complete

🚨 AI-GENERATED IMAGE

This appears to be synthetic content.

Confidence: 96.4%

🖼️ Image Analysis

\- Real Image: 3.6%

\- AI-Generated: 96.4%

💡 Recommendation

High confidence this is AI-generated. Be cautious about

context in which this image is shared.

\`\`\`

\#### 3. Multimodal Analysis

\*\*Combine text + image for comprehensive detection:\*\*

\`\`\`

User: "Miracle cure discovered!" + \[AI image\]

TruthLens AI:

🔍 Analysis Complete

🚨 HIGH RISK - Likely Fake/Synthetic

Analysis Mode: Text + Image (Multimodal)

Confidence: 94.3%

Combined Risk Score: 94.3%

📝 Text Analysis

\- Fake News: 92.3%

🖼️ Image Analysis

\- AI-Generated: 96.4%

💡 Recommendation

Both text and image show strong indicators of misinformation.

\`\`\`

\---

\### Interactive Commands

Type these commands in the chat:

| Command | Description |

|---------|-------------|

| \`help\` | Full usage guide |

| \`about\` | System architecture and details |

| \`stats\` | Performance metrics and benchmarks |

| \`hi\` or \`hello\` | Friendly greeting and introduction |

| \`thank you\` | Acknowledgment |

\*\*Example:\*\*

\`\`\`

User: "help"

TruthLens AI:

📖 How to Use TruthLens AI

For Text Analysis:

Just paste any news headline, social media post, or article excerpt.

I'll analyze it for:

\- Sensational language

\- Clickbait patterns

\- Emotional manipulation

\- Fake news indicators

\[... full help guide ...\]

\`\`\`

\---

\### Sample Buttons

Click pre-loaded samples for instant analysis:

\*\*🚨 Fake Samples:\*\*

\- "BREAKING: Scientists discover miracle cure that works 100%!!!"

\- "You won't BELIEVE what this celebrity said - SHOCKING revelation!!!"

\*\*✅ Real Samples:\*\*

\- "Senate committee schedules hearing on healthcare reform for next month"

\- "Research team publishes peer-reviewed findings in Nature journal"

These demonstrate the system's detection capabilities instantly.

\---

\## 💻 Programmatic Usage

\### Text Classification

\#### Basic Usage

\`\`\`python

from src.text\_classifier import TextClassifier

\# Initialize classifier

classifier = TextClassifier(

model\_path='models/text\_models/bert\_best\_model.pt',

device='cuda' # or 'cpu'

)

\# Analyze single headline

headline = "Scientists discover new planet in distant galaxy"

result = classifier.predict(headline)

print(f"Prediction: {result\['prediction'\]}") # REAL or FAKE

print(f"Confidence: {result\['confidence'\]:.1f}%") # 88.5%

print(f"Real Prob: {result\['real\_prob'\]:.3f}") # 0.885

print(f"Fake Prob: {result\['fake\_prob'\]:.3f}") # 0.115

\`\`\`

\#### Batch Processing

\`\`\`python

\# Analyze multiple headlines

headlines = \[

"BREAKING: Miracle cure found!!!",

"Senate passes new healthcare bill",

"You won't believe what happened next!"

\]

results = classifier.predict\_batch(headlines)

for headline, result in zip(headlines, results):

print(f"\\nHeadline: {headline}")

print(f"Verdict: {result\['prediction'\]} ({result\['confidence'\]:.1f}%)")

\`\`\`

\#### Get Detailed Analysis

\`\`\`python

\# Get full analysis with pattern detection

result = classifier.analyze(headline, include\_patterns=True)

print(f"Prediction: {result\['prediction'\]}")

print(f"Confidence: {result\['confidence'\]:.1f}%")

print(f"Patterns detected: {result\['patterns'\]}")

print(f"Recommendation: {result\['recommendation'\]}")

\`\`\`

\---

\### Image Classification

\#### Basic Usage

\`\`\`python

from src.image\_classifier import ImageClassifier

\# Initialize classifier

classifier = ImageClassifier(

model\_path='models/image\_models/best\_efficientnet\_cifake\_fast.pth',

device='cuda'

)

\# Analyze image from file path

result = classifier.predict("path/to/image.jpg")

print(f"Prediction: {result\['prediction'\]}") # REAL or AI-GENERATED

print(f"Confidence: {result\['confidence'\]:.1f}%") # 96.4%

print(f"Real Prob: {result\['real\_prob'\]:.3f}") # 0.036

print(f"AI Prob: {result\['ai\_prob'\]:.3f}") # 0.964

\`\`\`

\#### Analyze PIL Image

\`\`\`python

from PIL import Image

\# Load image

img = Image.open("photo.jpg")

\# Analyze

result = classifier.predict(img)

print(f"Verdict: {result\['prediction'\]}")

\`\`\`

\#### Batch Image Processing

\`\`\`python

import os

\# Analyze all images in a folder

image\_folder = "path/to/images/"

image\_files = \[os.path.join(image\_folder, f) for f in os.listdir(image\_folder)

if f.endswith(('.jpg', '.png', '.jpeg'))\]

results = classifier.predict\_batch(image\_files)

\# Count AI-generated vs real

ai\_count = sum(1 for r in results if r\['prediction'\] == 'AI-GENERATED')

print(f"AI-generated images: {ai\_count}/{len(results)}")

\`\`\`

\---

\### Multimodal Analysis (Planned Feature)

\`\`\`python

from src.multimodal\_fusion import MultimodalClassifier

\# Initialize multimodal system

classifier = MultimodalClassifier(

text\_model\_path='models/text\_models/bert\_best\_model.pt',

image\_model\_path='models/image\_models/best\_efficientnet\_cifake\_fast.pth',

text\_weight=0.6,

image\_weight=0.4

)

\# Analyze text + image

result = classifier.predict(

text="Miracle cure discovered!",

image="path/to/image.jpg"

)

print(f"Combined Verdict: {result\['prediction'\]}")

print(f"Combined Confidence: {result\['confidence'\]:.1f}%")

print(f"Text Analysis: {result\['text\_analysis'\]}")

print(f"Image Analysis: {result\['image\_analysis'\]}")

\`\`\`

\*\*Note:\*\* Multimodal fusion is designed but not yet deployed. See Phase 2 roadmap.

\---

\## 📓 Jupyter Notebooks

\### Running Notebooks

\*\*Launch Jupyter:\*\*

\`\`\`bash

\# Install Jupyter (if not already installed)

pip install jupyter notebook

\# Launch notebook server

jupyter notebook

\# Browser opens automatically

\# Navigate to notebooks/ folder

\`\`\`

\### Available Notebooks

\#### Text Classification

\*\*1. EDA & Visualization\*\* (\`01\_EDA\_visualizations.ipynb\`)

\- Explore dataset statistics

\- Visualize class distributions

\- Analyze text length patterns

\- Word frequency analysis

\*\*2. Baseline Models\*\* (\`02\_Baseline\_Model\_Training.ipynb\`)

\- Train 4 classical ML baselines

\- Compare SVM, Random Forest, Naive Bayes, Logistic Regression

\- Generate performance comparison plots

\*\*3. BERT Training\*\* (\`03\_BERT\_Training.ipynb\`)

\- Fine-tune BERT on Fakeddit dataset

\- Track training progress

\- Evaluate on test set

\- Save best model checkpoint

\*\*4. Dataset Setup\*\* (\`04\_Fakeddit\_Setup.ipynb\`)

\- Download and prepare Fakeddit dataset

\- Split into train/val/test

\- Verify data integrity

\*\*5. Chatbot Demo\*\* (\`05\_TruthLens\_Chatbot.ipynb\`)

\- Interactive chatbot interface

\- Real-time headline analysis

\- Sample testing

\#### Image Classification

\*\*1. Dataset Preparation\*\* (\`01\_CIFAKE\_Dataset\_Preparation.ipynb\`)

\- Download CIFAKE from Kaggle

\- Visualize sample images

\- Create train/val/test splits

\- Set up PyTorch data loaders

\*\*2. Model Training\*\* (\`02\_EfficientNet\_Training.ipynb\`)

\- Train EfficientNet-B0

\- Track training metrics

\- Implement early stopping

\- Save best checkpoint

\*\*3. Comprehensive Evaluation\*\* (\`03\_Model\_Evaluation.ipynb\`)

\- Detailed performance analysis

\- Confusion matrix

\- ROC curve (AUC: 0.9971)

\- Precision-Recall curve

\- Confidence calibration

\- Error analysis

\---

\## 🎯 Advanced Usage

\### Custom Model Training

\#### Train Text Model on Custom Dataset

\`\`\`python

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

\# Your custom dataset

train\_texts = \["headline 1", "headline 2", ...\]

train\_labels = \[0, 1, ...\] # 0=real, 1=fake

\# Tokenize

tokenizer = BertTokenizer.from\_pretrained('bert-base-uncased')

encodings = tokenizer(train\_texts, truncation=True, padding=True, max\_length=128)

\# Training arguments

training\_args = TrainingArguments(

output\_dir='./custom\_model',

num\_train\_epochs=2,

per\_device\_train\_batch\_size=16,

learning\_rate=2e-5,

logging\_steps=100,

save\_strategy='epoch'

)

\# Train

model = BertForSequenceClassification.from\_pretrained('bert-base-uncased', num\_labels=2)

\# ... setup trainer and train

\`\`\`

\### Fine-Tune Image Model

\`\`\`python

from torchvision import models

import torch.nn as nn

\# Load pre-trained EfficientNet

model = models.efficientnet\_b0(pretrained=True)

\# Modify for your classes

num\_features = model.classifier\[1\].in\_features

model.classifier = nn.Sequential(

nn.Dropout(0.2),

nn.Linear(num\_features, 2) # 2 classes: real/fake

)

\# Train on your dataset

\# See notebooks/image\_classification/02\_EfficientNet\_Training.ipynb

\`\`\`

\---

\### Confidence Threshold Tuning

Adjust decision thresholds based on your use case:

\`\`\`python

\# Conservative (minimize false positives)

confidence\_threshold = 0.85

result = classifier.predict(text)

if result\['confidence'\] > confidence\_threshold:

verdict = result\['prediction'\]

else:

verdict = "UNCERTAIN - Manual review recommended"

\# Aggressive (catch more fake news, allow some false positives)

confidence\_threshold = 0.60

\`\`\`

\*\*Recommended thresholds:\*\*

\- \*\*Social Media Flagging:\*\* 0.70 (balanced)

\- \*\*News Verification:\*\* 0.85 (high confidence)

\- \*\*Research/Analysis:\*\* 0.50 (capture all signals)

\---

\### Export Results

\#### Save Analysis Results

\`\`\`python

import json

\# Analyze headline

result = classifier.predict(headline)

\# Save to file

with open('analysis\_result.json', 'w') as f:

json.dump(result, f, indent=2)

\# Or CSV for multiple analyses

import pandas as pd

results\_list = \[\]

for headline in headlines:

result = classifier.predict(headline)

results\_list.append({

'headline': headline,

'prediction': result\['prediction'\],

'confidence': result\['confidence'\],

'real\_prob': result\['real\_prob'\],

'fake\_prob': result\['fake\_prob'\]

})

df = pd.DataFrame(results\_list)

df.to\_csv('batch\_analysis.csv', index=False)

\`\`\`

\---

\## 📊 API Reference

\### TextClassifier

\`\`\`python

class TextClassifier:

def \_\_init\_\_(self, model\_path, device='cuda'):

"""

Initialize text classifier

Args:

model\_path (str): Path to BERT model checkpoint

device (str): 'cuda' or 'cpu'

"""

def predict(self, text):

"""

Analyze single text

Args:

text (str): News headline or article

Returns:

dict: {

'prediction': 'REAL' or 'FAKE',

'confidence': float (0-100),

'real\_prob': float (0-1),

'fake\_prob': float (0-1)

}

"""

def predict\_batch(self, texts):

"""

Analyze multiple texts

Args:

texts (list): List of headlines

Returns:

list: List of prediction dictionaries

"""

def analyze(self, text, include\_patterns=True):

"""

Detailed analysis with pattern detection

Args:

text (str): Headline to analyze

include\_patterns (bool): Include suspicious patterns

Returns:

dict: Detailed analysis with patterns and recommendations

"""

\`\`\`

\### ImageClassifier

\`\`\`python

class ImageClassifier:

def \_\_init\_\_(self, model\_path, device='cuda'):

"""

Initialize image classifier

Args:

model\_path (str): Path to EfficientNet checkpoint

device (str): 'cuda' or 'cpu'

"""

def predict(self, image):

"""

Analyze single image

Args:

image: PIL Image or file path

Returns:

dict: {

'prediction': 'REAL' or 'AI-GENERATED',

'confidence': float (0-100),

'real\_prob': float (0-1),

'ai\_prob': float (0-1)

}

"""

def predict\_batch(self, images):

"""

Analyze multiple images

Args:

images (list): List of PIL Images or paths

Returns:

list: List of prediction dictionaries

"""

\`\`\`

\---

\## 🎓 Example Use Cases

\### Use Case 1: Social Media Monitoring

\*\*Monitor Twitter feed for fake news:\*\*

\`\`\`python

import tweepy # You'd need to install this

from src.text\_classifier import TextClassifier

\# Initialize

classifier = TextClassifier(model\_path='models/text\_models/bert\_best\_model.pt')

\# Fetch tweets (pseudo-code)

tweets = fetch\_recent\_tweets(hashtag="#breaking")

\# Analyze each

flagged = \[\]

for tweet in tweets:

result = classifier.predict(tweet.text)

if result\['prediction'\] == 'FAKE' and result\['confidence'\] > 80:

flagged.append({

'tweet': tweet,

'confidence': result\['confidence'\]

})

print(f"Flagged {len(flagged)} suspicious tweets")

\`\`\`

\### Use Case 2: News Article Verification

\*\*Check article headlines before sharing:\*\*

\`\`\`python

from src.text\_classifier import TextClassifier

classifier = TextClassifier(model\_path='models/text\_models/bert\_best\_model.pt')

\# Article headline

headline = "New study reveals shocking findings about coffee"

result = classifier.analyze(headline, include\_patterns=True)

if result\['confidence'\] > 85:

print(f"HIGH CONFIDENCE: {result\['prediction'\]}")

print(f"Patterns: {result\['patterns'\]}")

print(f"Recommendation: {result\['recommendation'\]}")

else:

print("UNCERTAIN - Verify through additional sources")

\`\`\`

\### Use Case 3: Image Authenticity Verification

\*\*Check if viral image is AI-generated:\*\*

\`\`\`python

from src.image\_classifier import ImageClassifier

classifier = ImageClassifier(model\_path='models/image\_models/best\_efficientnet\_cifake\_fast.pth')

\# Viral image circulating on social media

result = classifier.predict("viral\_image.jpg")

if result\['prediction'\] == 'AI-GENERATED' and result\['confidence'\] > 90:

print("⚠️ WARNING: High confidence this is AI-generated")

print(f"Confidence: {result\['confidence'\]:.1f}%")

print("Recommendation: Verify source before sharing")

else:

print("✅ Appears to be authentic photograph")

\`\`\`

\### Use Case 4: Batch Dataset Analysis

\*\*Analyze entire dataset:\*\*

\`\`\`python

import pandas as pd

from src.text\_classifier import TextClassifier

\# Load dataset

df = pd.read\_csv('news\_dataset.csv')

\# Initialize

classifier = TextClassifier(model\_path='models/text\_models/bert\_best\_model.pt')

\# Analyze all

predictions = \[\]

for headline in df\['headline'\]:

result = classifier.predict(headline)

predictions.append(result\['prediction'\])

\# Add to dataframe

df\['ai\_prediction'\] = predictions

\# Save

df.to\_csv('analyzed\_dataset.csv', index=False)

\# Statistics

print(f"Fake news detected: {sum(p == 'FAKE' for p in predictions)}")

print(f"Real news detected: {sum(p == 'REAL' for p in predictions)}")

\`\`\`

\---

\## 🧪 Testing & Validation

\### Test Model Loading

\`\`\`python

\# Test BERT loading

from src.text\_classifier import TextClassifier

try:

classifier = TextClassifier(model\_path='models/text\_models/bert\_best\_model.pt')

print("✅ BERT model loaded successfully")

except Exception as e:

print(f"❌ Error loading BERT: {e}")

\# Test EfficientNet loading

from src.image\_classifier import ImageClassifier

try:

classifier = ImageClassifier(model\_path='models/image\_models/best\_efficientnet\_cifake\_fast.pth')

print("✅ EfficientNet model loaded successfully")

except Exception as e:

print(f"❌ Error loading EfficientNet: {e}")

\`\`\`

\### Test Inference

\`\`\`python

\# Quick inference test

classifier = TextClassifier(model\_path='models/text\_models/bert\_best\_model.pt')

test\_cases = \[

("Real news test", "Senate schedules hearing"),

("Fake news test", "MIRACLE CURE FOUND!!!")

\]

for name, text in test\_cases:

result = classifier.predict(text)

print(f"{name}: {result\['prediction'\]} ({result\['confidence'\]:.1f}%)")

\`\`\`

\*\*Expected Output:\*\*

\`\`\`

Real news test: REAL (87.2%)

Fake news test: FAKE (93.5%)

\`\`\`

\---

\## 📈 Performance Monitoring

\### Track Inference Time

\`\`\`python

import time

start = time.time()

result = classifier.predict(headline)

elapsed = time.time() - start

print(f"Inference time: {elapsed\*1000:.1f}ms")

\# Expected:

\# GPU: 300-500ms

\# CPU: 1500-2500ms

\`\`\`

\### Memory Usage

\`\`\`python

import torch

\# Check GPU memory

if torch.cuda.is\_available():

print(f"GPU memory allocated: {torch.cuda.memory\_allocated()/1e9:.2f} GB")

print(f"GPU memory reserved: {torch.cuda.memory\_reserved()/1e9:.2f} GB")

\# Clear cache if needed

torch.cuda.empty\_cache()

\`\`\`

\---

\## 🎨 Customization

\### Modify Chatbot Appearance

Edit \`src/chatbot.py\`:

\`\`\`python

\# Change theme

with gr.Blocks(theme=gr.themes.Monochrome()) as demo: # Try: Soft, Glass, Base

\# Custom CSS

custom\_css = """

.gradio-container {

font-family: Arial, sans-serif;

max-width: 1200px;

}

"""

with gr.Blocks(css=custom\_css) as demo:

\`\`\`

\### Add Custom Patterns

Edit pattern detection in \`src/text\_classifier.py\`:

\`\`\`python

\# Add your own suspicious patterns

patterns = {

'Sensational': \['breaking', 'shocking', 'YOUR\_WORDS'\],

'Custom\_Category': \['word1', 'word2', 'word3'\]

}

\`\`\`

\### Adjust Fusion Weights

For multimodal analysis (when implemented):

\`\`\`python

\# In multimodal\_fusion.py

\# Adjust weights based on your trust in each model

w\_text = 0.7 # Increase if text model more reliable for your use case

w\_image = 0.3 # Decrease accordingly

\`\`\`

\---

\## 📱 Deployment Options

\### Option 1: Local Deployment

\`\`\`bash

\# Run locally

python src/chatbot.py

\# Access at: http://127.0.0.1:7860

\`\`\`

\### Option 2: Public Deployment (Gradio Share Link)

\`\`\`python

\# In chatbot.py

demo.launch(share=True) # Creates public URL for 72 hours

\`\`\`

\### Option 3: HuggingFace Spaces

\`\`\`bash

\# Deploy to HuggingFace for free hosting

gradio deploy

\# Follow prompts

\`\`\`

\### Option 4: Cloud Deployment

\- \*\*AWS/GCP/Azure:\*\* Use container deployment

\- \*\*Heroku:\*\* Add Procfile

\- \*\*Streamlit Cloud:\*\* Convert to Streamlit (alternative to Gradio)

\---

\## 🐛 Common Issues & Solutions

\### Issue: "Prediction always returns REAL"

\*\*Cause:\*\* Model not loaded correctly or wrong checkpoint

\*\*Solution:\*\*

\`\`\`python

\# Verify model file

import torch

checkpoint = torch.load('models/text\_models/bert\_best\_model.pt')

print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "State dict loaded")

\# Ensure correct loading

model.load\_state\_dict(checkpoint) # Not: model.load\_state\_dict(checkpoint\['model\_state\_dict'\])

\`\`\`

\### Issue: "Gradio interface won't load"

\*\*Solution:\*\*

\`\`\`bash

\# Try different port

python src/chatbot.py --port 7861

\# Or in code:

demo.launch(server\_port=7861)

\`\`\`

\### Issue: "Out of memory error"

\*\*Solution:\*\*

\`\`\`python

\# Use CPU instead

device = 'cpu'

\# Or reduce batch size

batch\_size = 8 # Instead of 16 or 32

\`\`\`

\---

\## 💡 Best Practices

\### For Accurate Results

1\. ✅ \*\*Complete headlines:\*\* Provide full headlines, not fragments

2\. ✅ \*\*Original text:\*\* Don't modify or paraphrase

3\. ✅ \*\*Clear images:\*\* Upload high-quality images when possible

4\. ✅ \*\*Context matters:\*\* Consider source and publication date

5\. ✅ \*\*Verify important claims:\*\* Always fact-check critical information

\### For Performance

1\. ✅ \*\*Use GPU:\*\* 5-10× faster than CPU

2\. ✅ \*\*Batch processing:\*\* Analyze multiple items together

3\. ✅ \*\*Cache results:\*\* Don't re-analyze identical content

4\. ✅ \*\*Close other programs:\*\* Free up GPU memory

\### For Production Use

1\. ✅ \*\*Set confidence thresholds:\*\* Based on your risk tolerance

2\. ✅ \*\*Human-in-the-loop:\*\* Review uncertain predictions

3\. ✅ \*\*Monitor performance:\*\* Track accuracy over time

4\. ✅ \*\*Update regularly:\*\* Retrain on new data periodically

\---

\## 📚 Further Reading

\- \*\*Full Report:\*\* \[docs/Final\_Project\_Report.pdf\](docs/Final\_Project\_Report.pdf)

\- \*\*Setup Guide:\*\* \[docs/SETUP.md\](docs/SETUP.md)

\- \*\*Dataset Info:\*\* \[docs/dataset\_info.md\](docs/dataset\_info.md)

\### External Resources

\- \*\*BERT Paper:\*\* \[Devlin et al., 2019\](https://arxiv.org/abs/1810.04805)

\- \*\*Fakeddit Dataset:\*\* \[Yang & Shu, 2020\](https://arxiv.org/abs/1911.03854)

\- \*\*EfficientNet Paper:\*\* \[Tan & Le, 2019\](https://arxiv.org/abs/1905.11946)

\- \*\*CIFAKE Dataset:\*\* \[Bird & Lotfi, 2024\](https://ieeexplore.ieee.org/document/10423388)

\---

\## ❓ FAQ

\*\*Q: Can I use this commercially?\*\*

A: Yes! MIT License allows commercial use. See LICENSE file.

\*\*Q: Do I need a GPU?\*\*

A: No, but recommended. CPU works but is slower (2-3 seconds vs. 0.5 seconds).

\*\*Q: What languages are supported?\*\*

A: Currently English only. Multilingual expansion planned for Phase 2.

\*\*Q: How accurate is it?\*\*

A: Text: 84.5%, Image: 97.66%. See performance section in README.

\*\*Q: Can I train on my own dataset?\*\*

A: Yes! See "Custom Model Training" section above.

\*\*Q: How do I report bugs?\*\*

A: Create an issue on GitHub with error details.

\*\*Q: Is the image model deployed in the chatbot?\*\*

A: Not yet. Currently text-only. Image integration planned for Phase 2.

\*\*Q: Can I integrate this into my app?\*\*

A: Yes! Use the programmatic API (see examples above).

\---

\## 🤝 Contributing

Found a bug or want to improve something?

1\. Fork the repository

2\. Create feature branch (\`git checkout -b feature/improvement\`)

3\. Make your changes

4\. Test thoroughly

5\. Submit pull request

See \[CONTRIBUTING.md\](CONTRIBUTING.md) for detailed guidelines.

\---

\## 📞 Support

\*\*Need help?\*\*

1\. ✅ Check this usage guide

2\. ✅ Review \[SETUP.md\](SETUP.md) for installation issues

3\. ✅ Search \[GitHub Issues\](https://github.com/YOUR\_USERNAME/TruthLens-AI/issues)

4\. ✅ Create new issue with details

\*\*For academic inquiries:\*\*

\- Course: EAI6010 Applications of AI

\- Institution: Northeastern University

\- Contact: \[Team contact information\]

\---

\*\*Happy Detecting! 🔍\*\*

\[⬅️ Back to README\](../README.md) | \[📖 Setup Guide\](SETUP.md) | \[📄 Full Report\](Final\_Project\_Report.pdf)

\---

\*Last Updated: January 2026\*

\*Version: 1.0\*