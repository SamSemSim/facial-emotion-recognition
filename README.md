# Facial Emotion Recognition Project

A real-time facial emotion recognition system using PyTorch and OpenCV. The system can detect faces and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Features

- Real-time facial emotion detection using webcam
- Deep learning model built with PyTorch
- Face detection using OpenCV
- Support for 7 different emotions
- Confidence scores for predictions
- Model achieves ~71.93% validation accuracy
- CUDA support for GPU acceleration
- Debug window showing all emotion probabilities

## Overview

This project implements a facial emotion recognition system that:
- Uses a deep learning model to detect emotions in real-time
- Processes webcam feed to detect faces
- Classifies emotions with confidence scores
- Provides an easy-to-use interface with debug information
- Utilizes GPU acceleration when available
- Implements smart emotion selection logic for better real-world performance

## Dataset Structure

The project requires a dataset organized as follows:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### Dataset Requirements

Each image in the dataset should be:
- Grayscale format
- 48x48 pixels
- Face centered in the frame

You can use datasets like:
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (Facial Expression Recognition Challenge)
- [CK+](http://www.jeffcohn.net/Resources/) Dataset
- Or create your own dataset following the structure above

## Model Architecture

The emotion recognition model uses a CNN with:
- 4 Convolutional blocks with batch normalization
- ReLU activation functions
- Max pooling layers for feature reduction
- Dropout layers (0.5) for regularization
- Fully connected layers for final classification
- Smart post-processing for more stable predictions

Specifications:
- Input: 48x48 grayscale images
- Output: 7 emotion classes
- Training accuracy: 71.93%
- Uses data augmentation for better generalization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SamSemSim/facial-emotion-recognition.git
cd facial-emotion-recognition
```

2. Create a virtual environment (Python 3.10 required):
```bash
# Windows
py -3.10 -m venv emotion_env

# Linux/Mac
python3.10 -m venv emotion_env
```

3. Activate the virtual environment:
```bash
# Windows
.\emotion_env\Scripts\activate

# Linux/Mac
source emotion_env/bin/activate
```

4. Install PyTorch with CUDA support (Windows):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download one of the recommended datasets (e.g., FER2013)
2. Create a `dataset` folder in the project root
3. Organize images into train/test splits as shown in the structure above
4. Ensure all images are in grayscale and 48x48 pixels

## Usage

### Training the Model

```bash
python train_model.py
```

The training script will:
- Load the dataset from the `dataset` folder
- Train for 100 epochs
- Save the best model based on validation accuracy
- Show training progress and metrics
- Apply learning rate scheduling for better convergence

### Real-time Emotion Detection

```bash
python webcam_emotion.py
```

Features:
- Press 'q' to quit the application
- Main window shows:
  - Detected faces with bounding boxes
  - Predicted emotion
  - Confidence score
- Debug window shows:
  - Probabilities for all emotions
  - Real-time updates

## Training Results

The current model achieves:
- Validation accuracy: 71.93%
- Training completed in 100 epochs
- Uses learning rate scheduling
- Implements data augmentation
- Smart post-processing for better real-world performance

## Requirements

- Python 3.10
- PyTorch 2.5.1 (with CUDA 11.8 support)
- OpenCV 4.10.0
- NumPy 1.26.3
- Pillow 10.2.0
- Matplotlib 3.10.0
- CUDA Toolkit 11.8 (for GPU acceleration)

See `requirements.txt` for all dependencies.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the PyTorch and OpenCV communities
- Dataset providers and researchers in emotion recognition
- Contributors and users of this project
