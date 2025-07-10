# Pneumonia Detection using Inception-V3

This project fine-tunes a pre-trained **Inception-V3** model on the **PneumoniaMNIST** dataset to classify chest X-ray images as either **Normal** or **Pneumonia**.

---

## 📁 Dataset

- **Name**: PneumoniaMNIST  
- **Source**: [Kaggle - PneumoniaMNIST](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data)  
- **Format**: `.npy` files:
  - `train_images.npy`, `train_labels.npy`  
  - `val_images.npy`, `val_labels.npy`  
  - `test_images.npy`, `test_labels.npy`

---

## ⚙️ Setup Instructions

Install the required libraries using:

```bash
pip install -r requirements.txt
```

---

##  How to Run the Project

1. Download the PneumoniaMNIST dataset as `.npy` files  
2. Place all `.npy` files in your project directory  
3. Open and run `Pneumonia_Inception_V3.ipynb` notebook ( Colab)  
4. Training will run for 5 epochs and print evaluation results

---

##  Evaluation Metrics (on Test Set)

```
              precision    recall  f1-score   support

      Normal       0.92      0.71      0.80       234
   Pneumonia       0.85      0.96      0.90       390

    accuracy                           0.87       624
   macro avg       0.88      0.84      0.85       624
weighted avg       0.87      0.87      0.86       624
```

**Insights:**
- Model achieved high **recall (96%) for Pneumonia**, making it suitable for early diagnosis  
- Performance for Normal cases improved significantly after handling class imbalance  
- Balanced F1-scores across both classes

---

##  Model Details

- **Architecture**: Inception-V3 from `torchvision.models`  
- **Modifications**: Final FC layer changed to output 2 classes  
- **Input Size**: 299x299 images (grayscale → RGB)  
- **Augmentation**:
  - `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`  
- **Loss Function**: `CrossEntropyLoss` with class weights to handle imbalance

---

##  Hyperparameter Choices

| Parameter      | Value   | Reason                                                    |
|----------------|---------|-----------------------------------------------------------|
| Learning Rate  | 0.001   | Standard rate for Adam optimizer                          |
| Batch Size     | 32      | Balanced memory usage and training speed                  |
| Epochs         | 5       | Sufficient to achieve convergence with pretrained model   |
| Optimizer      | Adam    | Efficient for deep learning and adaptive learning rates   |
| Loss Function  | CrossEntropyLoss with weights | To penalize under-represented class |

---

##  Files in This Project

```
Pneumonia_Inception_V3.ipynb     # Complete notebook with code and results  
requirements.txt                 # Required packages for environment setup  
README.md                        # Instructions, results, and insights (this file)
```

---

##  Summary

This project demonstrates how to fine-tune a pre-trained CNN (Inception-V3) for medical image classification. It also explores the impact of class imbalance and how to address it using **data augmentation** and **weighted loss functions**.

---


