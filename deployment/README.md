# Emotion Detection Deployment

Deployment package for the Emotion Detection in Crisis Social Media project.

## Components

- Dockerfile
- Python inference application
- Trained tokenizer files
- Model configuration files

## Technologies

- Python
- Transformers
- PyTorch
- Docker

## Run

```bash
pip install -r requirements.txt
python app.py
```

## Container Build

```bash
docker build -t emotion-detection .
docker run -p 5000:5000 emotion-detection
```

## Note

The trained model weights are excluded from the repository because of their large size.