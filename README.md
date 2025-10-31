# Lung Cancer Detection

*Automated detection of lung cancer nodules from CT scans using DenseNet121.*

---

## Project Overview

Lung cancer is one of the most common and fatal cancers worldwide, largely because it is often diagnosed at advanced stages when treatment options are limited. Traditional detection methods rely heavily on radiologists manually examining CT scans, a process that is not only time-consuming but also prone to human error due to the sheer volume of images and the subtle nature of early-stage nodules. As medical imaging datasets continue to grow, there is an urgent need for **automated, accurate, and scalable diagnostic tools** that can assist clinicians in making timely decisions.  

This project was developed to address that challenge by building a **deep learning pipeline for automated lung cancer detection**. The goal is to create a system that can consistently identify cancerous nodules from CT scans with high sensitivity, reducing the chances of missed diagnoses. By leveraging **DenseNet121 with transfer learning**, the model is able to extract complex visual patterns even from a relatively limited dataset, which is common in the medical field.  

To further improve robustness, the pipeline incorporates advanced preprocessing steps, realistic data augmentation strategies, and techniques to handle class imbalance. This ensures that the system not only achieves good performance on training data but also generalizes well to unseen cases. The resulting framework is **modular, reproducible, and clinically relevant**. Ultimately, this project aims to contribute toward **early cancer detection**, improved diagnostic efficiency, and the long-term vision of using AI to support healthcare professionals in saving lives.  

---

## Setup

Clone the repository and set up your environment:  

```bash
git clone https://github.com/samikshapatel27/LungCancerDetection.git
cd LungCancerDetection
pip install -r requirements.txt
```

Place raw CT scans in `.mhd` format under:  

```
../src/data/raw/
```

Run the notebook to extract sub-images, normalize, augment, train the DenseNet121 model, and evaluate performance. The final model is saved as:

```
best_densenet.keras
```

---

## Methodology  

### **Data Preprocessing**  
The pipeline begins with careful preprocessing of raw CT scans to ensure that the input to the model is both standardized and informative. From each 3D scan, smaller **50×50 voxel sub-images** are extracted around the annotated nodule regions, focusing the model on the most clinically relevant structures. Since CT images are measured in **Hounsfield Units (HU)**, all voxel values are normalized to a consistent range of `[-1000, 400]`, which captures the intensity spectrum relevant for lung tissue and nodules while suppressing irrelevant background. Although CT images are naturally grayscale, they are **converted into RGB format** to be compatible with the pretrained DenseNet121 architecture, which was originally designed for color image inputs. This preprocessing step ensures consistency, reduces noise, and prepares the dataset for efficient training.  

### **Data Augmentation**  
To enhance generalization and prevent overfitting, the dataset undergoes a comprehensive **augmentation pipeline**. Techniques such as random rotations, spatial shifts, shear transformations, zooming, and horizontal flips are applied to mimic real-world variability in scan orientations and patient anatomy. In addition, **Gaussian noise** is injected to simulate imaging artifacts and scanner variability, further improving robustness. Since lung cancer datasets are typically imbalanced, with far fewer positive (cancerous) nodules than negative ones, **positive samples are oversampled and augmented multiple times**. This strategy helps the model maintain high sensitivity to cancerous findings without being biased toward the majority class.  

### **Model Training**  
At the core of the system lies **DenseNet121**, a deep convolutional neural network known for its dense connectivity, which promotes feature reuse and efficient gradient flow. The pretrained DenseNet121 backbone is fine-tuned and extended with additional **fully connected dense layers**, along with **batch normalization** for stability and **dropout layers** to reduce overfitting. The final output layer uses a **sigmoid activation**, producing a probability score for binary classification (cancerous vs. non-cancerous nodule).  

Training is optimized with the **Adam optimizer**, chosen for its adaptability and strong performance in deep learning tasks. The model’s performance is continuously evaluated using metrics such as **Accuracy, AUC (Area Under Curve), Precision, and Recall**, ensuring both balanced performance and sensitivity to true positives. To improve convergence and avoid overfitting, training employs callbacks including **EarlyStopping** (to halt training when no further improvement is observed) and **ReduceLROnPlateau** (to lower the learning rate dynamically when progress stalls). Together, these strategies ensure efficient training and robust model performance across different test conditions.  

---

## Model Outputs  

The model achieved 90% overall accuracy with ~0.95 AUC, strong cancer recall (87%), balanced precision-recall tradeoff, and stable training curves indicating good generalization.  

<img src="result/performance.png" width="50%" />  

<img src="result/confusion_matrix.png" width="70%" />  

<p float="left">
  <img src="result/learning_curve_auc.png" width="45%" />  
  <img src="result/learning_curve_loss.png" width="45%" />  
</p>

---

## References

* **Data Source:** [LUNA16 Dataset – CT Scan Images](https://luna16.grand-challenge.org/Data/)  
* Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks*. CVPR.  
* Medical image augmentation techniques adapted from [nnU-Net](https://arxiv.org/abs/1809.10486) and [MedicalTorch](https://github.com/perone/medicaltorch).

