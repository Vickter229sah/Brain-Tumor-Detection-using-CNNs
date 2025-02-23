### Summary:
This project demonstrates the use of Convolutional Neural Networks (CNNs) to detect brain tumors from MRI images. Using TensorFlow/Keras, OpenCV, Matplotlib, and NumPy, we preprocess the images, normalize them, and augment the training data. The CNN model consists of three convolutional layers followed by max pooling and dense layers for final classification. Trained on a dataset of MRI images, the model achieves good accuracy in distinguishing between tumor and non-tumor images. The performance is visualized through training and validation accuracy plots. This project provides a foundational approach for brain tumor detection, with potential for further enhancement through more complex architectures and advanced techniques.


### Project Note: Brain Tumor Detection using CNNs

#### Project Overview
This project aims to detect brain tumors from MRI images using Convolutional Neural Networks (CNNs). The model classifies images into two categories: tumor and normal.

#### Project Specifications
- **Programming Language**: Python
- **Libraries Used**:
  - TensorFlow/Keras: For building and training the CNN model
  - OpenCV: For image preprocessing
  - Matplotlib: For visualization
  - NumPy: For numerical operations
- **Dataset Structure**:
  - `dataset/train/tumor`: Contains training images with tumors
  - `dataset/train/normal`: Contains training images without tumors
  - `dataset/test/tumor`: Contains test images with tumors
  - `dataset/test/normal`: Contains test images without tumors
- **Image Size**: 150x150 pixels

#### Key Points

1. **Environment Setup**:
    - Install necessary libraries: TensorFlow, OpenCV, Matplotlib, and NumPy using `pip install`.

2. **Data Loading and Preprocessing**:
    - Use `ImageDataGenerator` for data augmentation and normalization.
    - Load images from directories and preprocess them by rescaling pixel values to [0, 1].

3. **CNN Model Architecture**:
    - **Input Layer**: 150x150x3 images.
    - **Convolutional Layers**: Three layers with increasing filters (32, 64, 128), each followed by a MaxPooling layer.
    - **Flatten Layer**: Converts 2D matrices to a 1D vector.
    - **Dense Layers**: Two fully connected layers, the first with 512 neurons and ReLU activation, the second with 1 neuron and sigmoid activation for binary classification.

4. **Model Compilation**:
    - Optimizer: Adam
    - Loss Function: Binary Cross-Entropy
    - Metric: Accuracy

5. **Model Training**:
    - Train the model for 10 epochs.
    - Use training data for training and validation data for validation.

6. **Model Evaluation**:
    - Evaluate the model on the test dataset.
    - Plot training and validation accuracy over epochs for performance visualization.

7. **Making Predictions**:
    - Load and preprocess new images.
    - Use the trained model to predict if the image has a tumor.
    - Display prediction results.

8. **Visualization**:
    - Display sample images from the training set with class labels.
    - Plot training and validation accuracy to visualize the model's performance over time.

#### Steps to Run the Project

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Install Dependencies**:
    ```sh
    pip install tensorflow opencv-python matplotlib numpy
    ```

3. **Run the Training Script**:
    - Ensure the dataset is structured correctly.
    - Execute the Python script to train the model.

4. **Evaluate and Visualize Results**:
    - Evaluate the model on test data.
    - Visualize the accuracy plots and sample predictions.

5. **Make Predictions on New Images**:
    - Use the provided function to predict tumors on new MRI images.

#### Conclusion
This project showcases a basic implementation of a CNN for brain tumor detection using MRI images. It provides a foundation that can be further improved by experimenting with different architectures, data augmentation techniques, and more extensive training.

#### Future Work
- Experiment with deeper and more complex CNN architectures.
- Utilize advanced data augmentation techniques.
- Increase the number of epochs and batch size for training.
- Implement techniques like transfer learning for improved accuracy.
- Explore other metrics like precision, recall, and F1-score for a more comprehensive evaluation.

#### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

#### Acknowledgments
- TensorFlow and Keras for providing powerful libraries to build and train deep learning models.
- OpenCV and Matplotlib for image processing and visualization.
- The creators of the MRI dataset for making it publicly available.![Screenshot 2024-08-07 at 6 20 54 PM](https://github.com/user-attachments/assets/f7f16a43-9fbe-46cd-9593-d8e764b1260d)
![Screenshot 2024-08-07 at 6 21 00 PM](https://github.com/user-attachments/assets/168bd2c3-1ac4-4c56-a859-59cd6e868212)
![Screenshot 2024-08-07 at 6 21 00 PM](https://github.com/user-attachments/assets/5b846a95-bff8-4bba-959b-cedf09b663ca)![Screenshot 2024-08-07 at 6 21 12 PM](https://github.com/user-attachments/assets/e27e5a2f-84bf-4ae5-8413-e7c2f54bff40)

![Screenshot 2024-08-07 at 6 21 05 PM](https://github.com/user-attachments/assets/7a63e711-a738-425c-bc43-31cf8a336f88)
