# Facial Emotion Recognition Project

A real-time facial emotion recognition system using PyTorch and OpenCV. The system can detect faces and classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Features

- Real-time facial emotion detection using webcam
- Deep learning model built with PyTorch
- Face detection using OpenCV
- Support for 7 different emotions
- Confidence scores for predictions
- Model achieves ~68.7% validation accuracy

## Project Structure

```
├── dataset/
│   ├── train/          # Training dataset
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── test/           # Testing dataset
├── train_model.py      # Script for training the emotion recognition model
├── webcam_emotion.py   # Script for real-time emotion detection using webcam
└── requirements.txt    # Project dependencies
```

## Model Architecture

The emotion recognition model uses a Convolutional Neural Network (CNN) with:
- 4 Convolutional blocks with batch normalization and ReLU activation
- Max pooling layers for feature reduction
- Dropout for regularization
- Fully connected layers for classification
- Input: Grayscale images (48x48 pixels)
- Output: 7 emotion classes

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd facial-emotion-recognition
```

2. Create a virtual environment with Python 3.10:
```bash
python -m venv emotion_env
```

3. Activate the virtual environment:
- Windows:
```bash
.\emotion_env\Scripts\activate
```
- Linux/Mac:
```bash
source emotion_env/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (optional, pre-trained model available):
```bash
python train_model.py
```

2. Run real-time emotion detection:
```bash
python webcam_emotion.py
```

Controls:
- Press 'q' to quit the webcam application

## Training Results

The model achieves:
- Validation accuracy: 68.68%
- Training completed in 250 epochs
- Uses learning rate scheduling for optimization
- Implements data augmentation for better generalization

## Dataset

The dataset should be organized as follows:
- Training images in `dataset/train/[emotion_name]`
- Testing images in `dataset/test/[emotion_name]`
- Images should be grayscale and 48x48 pixels

## Requirements

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Pillow

See `requirements.txt` for specific versions.

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request #   f a c i a l - e m o t i o n - r e c o g n i t i o n  
 