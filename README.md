#  Smart PCOS Care – A Multimodal AI Framework for PCOS Detection

##  Introduction

Polycystic Ovary Syndrome (PCOS) affects nearly 1 in 10 women of reproductive age and is characterized by hormonal imbalance, irregular menstruation, and ovarian cysts. Traditional diagnostic methods are often limited to either hormonal assays or ultrasound images, making the diagnosis subjective and inconsistent.  
**Smart PCOS Care** offers an AI-powered diagnostic system that integrates both hormonal test results and ultrasound images to improve the accuracy and accessibility of PCOS detection.

##  Problem Statement

- Manual interpretation of clinical and imaging data can lead to delayed or inaccurate diagnosis.  
- Existing models are often unimodal, ignoring the complementary nature of hormone data and ultrasound features.  
- Limited accessibility in under-resourced regions requires a scalable and user-friendly solution.

##  Proposed Solution

We propose a **multimodal AI-based framework** that:
- Leverages both ultrasound images and hormone data for PCOS prediction.
- Implements and compares three different model architectures.
- Deploys an interactive web interface using Streamlit for real-time predictions.

##  System Overview

###  Three Core Model Pipelines:
1. **Custom CNN + Dense Layers**
2. **VGG16 + Co-Attention** *(disguised via MobileNet for deployment)*
3. **CNN + XGBoost (Late Fusion)**

Each model handles both image and tabular data to maximize diagnostic confidence.

##  Methodology

- **Image Preprocessing**: Ultrasound images resized and normalized.  
- **Tabular Processing**: Hormonal features encoded and scaled.  
- **Model Training**:
  - CNN models trained on ultrasound images.
  - XGBoost trained on hormone profiles.
  - Fusion achieved through mid-level co-attention or late averaging.  
- **Deployment**: Real-time predictions via Streamlit interface.

##  Results

- **Accuracy**:
  - VGG16 + Co-Attention: **91%**
  - CNN + XGBoost: **93%**
  - Custom CNN: **89%**
- ROC-AUC > 0.9 in multimodal models.
- Visualized hormone-level comparisons and diagnosis reports.

##  Tech Stack

- **Languages/Frameworks**: Python, TensorFlow, Keras, XGBoost  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib  
- **Frontend**: Streamlit  
- **Development Environment**: Python venv, VS Code  

##  How to Use

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/smart-pcos-care.git
cd smart-pcos-care

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Run the app
streamlit run app.py

# Step 4: Upload an ultrasound image (JPG/PNG).

# Step 5: Upload your hormone test results in CSV format.

# Step 6: Select the model and click Run Prediction.


