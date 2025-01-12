# Facial Emotion Recognition Project

A real-time facial emotion recognition system using PyTorch and OpenCV. The system can detect faces and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Features

- Real-time facial emotion detection using webcam
- Deep learning model built with PyTorch
- Face detection using OpenCV
- Support for 7 different emotions
- Confidence scores for predictions
- Model achieves ~68.7% validation accuracy

## Overview

This project implements a facial emotion recognition system that:
- Uses a deep learning model to detect emotions in real-time
- Processes webcam feed to detect faces
- Classifies emotions with confidence scores
- Provides an easy-to-use interface

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
- Dropout layers for regularization
- Fully connected layers for final classification

Specifications:
- Input: 48x48 grayscale images
- Output: 7 emotion classes
- Training accuracy: 68.68%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SamSemSim/facial-emotion-recognition.git
cd facial-emotion-recognition
```

2. Create a virtual environment (Python 3.10 recommended):
```bash
python -m venv emotion_env
```

3. Activate the virtual environment:
```bash
# Windows
.\emotion_env\Scripts\activate

# Linux/Mac
source emotion_env/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_model.py
```

The training script will:
- Load the dataset from the `dataset` folder
- Train for 250 epochs
- Save the best model based on validation accuracy
- Show training progress and metrics

### Real-time Emotion Detection

```bash
python webcam_emotion.py
```

Controls:
- Press 'q' to quit the application
- The application will show:
  - Detected faces with bounding boxes
  - Predicted emotion
  - Confidence score

## Training Results

The current model achieves:
- Validation accuracy: 68.68%
- Training completed in 250 epochs
- Uses learning rate scheduling
- Implements data augmentation for better generalization

## Requirements

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Pillow

See `requirements.txt` for specific versions.

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
