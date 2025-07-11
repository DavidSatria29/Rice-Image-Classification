# A Comparative Analysis of CNN Architectures for Rice Variety Identification

## 1. Project Overview
This project develops and evaluates Deep Learning models for the automated classification of rice varieties from image data. The analysis focuses on **Convolutional Neural Networks (CNNs)**, an algorithm well-suited for image recognition tasks.

The primary objective is to compare three different CNN architectures (AlexNet, VGG, and ResNet-50) to identify the most accurate and efficient model for this problem. This solution holds significant business value for the agricultural and food industries, particularly in improving the efficiency and accuracy of quality control (QC) processes.

## 2. Dataset
This analysis utilizes the **"Rice Image Dataset"** available on Kaggle. The dataset consists of 75,000 images, evenly distributed across five different rice varieties.

-   **Classes/Varieties:**
    1.  Arborio
    2.  Basmati
    3.  Ipsala
    4.  Jasmine
    5.  Karacadag
-   **Data Format:** Each image is a 250x250 pixel color photograph of a single grain of rice against a plain background.

## 3. Methodology
The project workflow is divided into several key stages:

### a. Data Preprocessing
1.  **Data Loading:** The dataset was downloaded from Kaggle and loaded using the Hugging Face `datasets` library.
2.  **Data Splitting:** The dataset was split into an 80% training set (60,000 images) and a 20% testing set (15,000 images).
3.  **Feature Engineering (Image Resizing):** The image size was significantly reduced from 250x250 to **32x32 pixels**. This decision was made because the primary distinguishing feature for rice is its shape, not fine-grained texture. This step drastically reduced the computational load.
4.  **Normalization:** Pixel values were scaled from the 0-255 range to a 0-1 range to help the model learn faster and more stably.
5.  **One-Hot Encoding:** The integer class labels (0-4) were converted into a one-hot binary vector format, which is required for the `categorical_crossentropy` loss function.

### b. Model Building and Training
Three different CNN architectures, adapted for the 32x32 pixel input, were built and trained:
1.  **Mini-AlexNet:** A simplified version of the classic AlexNet architecture.
2.  **Mini-VGG:** An architecture inspired by VGG, using stacked 3x3 convolutional layers.
3.  **ResNet-50:** A very deep and powerful architecture, trained from scratch.

All models were trained for 3 epochs with a `batch_size` of 128 and the Adam optimizer for a fair comparison.

## 4. Results
The training of all three models yielded highly competitive results, proving that the morphological features in the 32x32 images are sufficient for accurate classification.

### Peak Model Performance Summary
The following table summarizes the best performance achieved by each model on the validation data.

| Model Architecture | Best Validation Accuracy | Loss at Best Epoch | Best Epoch |
| :----------------- | :----------------------- | :------------------- | :----------- |
| AlexNet            | 0.9755                   | 0.0749               | 2            |
| VGG-16             | 0.9808                   | 0.0559               | 3            |
| ResNet-50          | 0.9803                   | 0.0622               | 3            |

## 5. Conclusion & Recommendation
Based on the evaluation results, the **Mini-VGG architecture is recommended as the final model**. Although ResNet-50 achieved a comparable accuracy, the Mini-VGG model offers the best balance of performance and efficiency. It achieved the highest validation accuracy (**98.08%**) with a significantly shorter training time compared to the much more complex ResNet-50.

## 6. How to Run
1.  Clone this repository.
2.  Ensure all libraries listed at the top of the Jupyter Notebook (`Rice_Image_Clasification_.ipynb`) are installed.
3.  Run the notebook cells sequentially from top to bottom.
4.  It is highly recommended to use an environment with a GPU accelerator (like in Google Colab) to speed up the training process.

## 7. Future Work
For future development, the following steps could be explored:
-   **Transfer Learning:** Use pre-trained weights (e.g., from ImageNet) for the ResNet-50 model to potentially improve accuracy and reduce training time.
-   **Data Augmentation:** Apply augmentation techniques like random rotations and flips to the training data to enhance model robustness.
-   **Explore Lightweight Architectures:** Test other efficient models like MobileNet or EfficientNet, which are ideal for deployment on resource-constrained devices.
