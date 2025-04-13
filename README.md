# CrowdSentry: 2025 AI Hackfest

## Overview

This project is designed to aid in event organizers who deal with high density crowds. There often simple issues with management and organization that can lead to uncesscary injuries and even deaths that could be avoided if the organizers were able to monitor the crowd in real time. 

---

## Features

- **Crowd Counting**: Trained on the ShanghaiTech Dataset this AI model can provide precise headcounts
- **Pre-trained Models**: Fine-tuned on dense and sparse crowd datasets.

---

## Datasets Used

- **ShanghaiTech Dataset**:
    - **Part A**: Dense urban crowd scenes.
    - **Part B**: Sparse suburban scenes.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/callmetan7/CrowdSentry.git
   cd CrowdSentry 
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
Download the ShanghaiTech Dataset and replicate this file structure
Link to download: https://www.kaggle.com/datasets/tthien/shanghaitech
   ``` 
   data/
   ├── ShanghaiTech/
   │   ├── part_A/
   │   │   ├── train_data/
   │   │   ├── test_data/
   │   ├── part_B/
   │   │   ├── train_data/
   │   │   ├── test_data/
   
   ```

4**Prepare Pre-trained Models**:
   Place your pre-trained `.pth` files in the `models/` directory.

---

## Usage

### **Run the Application**

Start the Streamlit app:

```bash
streamlit run app.py
```

### **Predict Crowd Count**

1. Upload a crowd image via the interface.
2. Select a pre-trained model from the dropdown menu.
3. Click **Predict Headcount** to view the predicted count and density map.

---

## Project Structure

```
deploy_model/
├── app.py                   # Main Streamlit application
├── main.py                  # Main preprocessing, training, and evaluation loop 
├── data/                    # Dataset 
│   ├── ShanghaiTech/
│   │   ├── part_A/
│   │   │   ├── train_data/
│   │   │   ├── test_data/
│   │   ├── part_B/
│   │   │   ├── train_data/
│   │   │   ├── test_data/
├── models/                  # Path to pre-trained models
├── src/                     # Source files
│   ├── model.py             # MCNN model architecture
│   ├── preprocess.py        # Data preprocessing utilities
│   ├── dataset.py           # Dataset loader
│   └── train.py             # Training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Dependencies

The project uses the following libraries:

- **PyTorch**: For training and inference.
- **Streamlit**: For building the user interface.
- **OpenCV**: For image preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib**: For density map visualization.

---
