# Plant Disease Prediction using Deep Learning Ensemble

## Overview
A deep learning-based plant disease classification system that combines multiple pre-trained convolutional neural networks (CNNs) to achieve high accuracy in identifying plant diseases from leaf images. The project utilizes an ensemble approach combining VGG19, InceptionV3, and ResNet50 models for robust disease detection.

## Features
- **Multi-Model Ensemble**: Combines three state-of-the-art CNN architectures (VGG19, InceptionV3, ResNet50)
- **Transfer Learning**: Leverages pre-trained ImageNet weights for better feature extraction
- **Data Augmentation**: Implements comprehensive image augmentation techniques to improve model generalization
- **High Accuracy**: Achieves up to **99.21% validation accuracy** on the test dataset
- **GPU Acceleration**: Optimized for CUDA-enabled GPU training

## Model Architecture
The ensemble model architecture consists of:
1. **VGG19**: Deep network with small 3x3 filters
2. **InceptionV3**: Multi-scale feature extraction with inception modules
3. **ResNet50**: Residual connections for deeper network training
4. **Fusion Layer**: Combines outputs from all three models through a fully connected layer

Each base model's pre-trained weights are frozen, and only the custom classification heads are trained.

## Dataset
- **Classes**: 4 different plant disease categories
- **Image Size**: 299x299 pixels (optimized for InceptionV3)
- **Data Split**: Training and validation sets
- **Augmentation**: Random rotation, horizontal flip, and resized cropping

## Performance
- **Best Validation Accuracy**: 99.21% (Epoch 16)
- **Final Training Accuracy**: 99.64%
- **Final Validation Accuracy**: 97.63%
- **Training Epochs**: 50

## Key Technologies
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and image transformations
- **CUDA**: GPU acceleration
- **OpenCV**: Image processing
- **Matplotlib**: Visualization

## Training Process
The model demonstrates excellent learning characteristics:
- Rapid initial convergence (89.33% accuracy in first epoch)
- Consistent improvement with minimal overfitting
- Stable validation performance across epochs
- Effective ensemble learning combining strengths of multiple architectures

## Data Augmentation Techniques
- Random rotation (±20 degrees)
- Random horizontal flipping
- Random resized cropping (scale: 0.8-1.0)
- ImageNet normalization

### Model Structure
```
EnsembleModel(
  ├── VGG19 (frozen backbone + custom classifier)
  ├── InceptionV3 (frozen backbone + custom classifier)  
  ├── ResNet50 (frozen backbone + custom classifier)
  └── Fusion Layer (3*num_classes → num_classes)
)
```

## Results Visualization
The training process shows:
- **Loss Curves**: Steady decrease in both training and validation loss
- **Accuracy Curves**: Consistent improvement reaching near-perfect classification
- **Model Stability**: Minimal overfitting with good generalization

## Future Improvements
- [ ] Add more plant species and disease types
- [ ] Implement grad-CAM for model interpretability
- [ ] Deploy as web application or mobile app
- [ ] Add real-time inference capabilities
- [ ] Experiment with additional ensemble techniques

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- Pre-trained models from torchvision
- Transfer learning techniques for efficient training
- Ensemble methods for improved accuracy
