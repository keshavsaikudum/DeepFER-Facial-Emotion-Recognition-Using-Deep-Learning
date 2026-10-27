# DeepFER: Project Report

## 1. Problem
Classify a face image into one of seven emotions (angry, disgust, fear, happy, neutral, sad,
surprise). The system should be accurate and fast enough for live video. The brief asks for a
CNN and transfer learning, with rotation, scaling and flipping augmentation, evaluation by
accuracy, precision, recall and F1, a real-time component and a user-facing application.

## 2. Data
The supplied archive is the FER2013 dataset: 48x48 grayscale face crops, already aligned.

| class | train | test |
|---|---:|---:|
| angry | 3,995 | 958 |
| disgust | 436 | 111 |
| fear | 4,097 | 1,024 |
| happy | 7,215 | 1,774 |
| neutral | 4,965 | 1,233 |
| sad | 4,830 | 1,247 |
| surprise | 3,171 | 831 |
| **total** | **28,709** | **7,178** |

- **Split:** 10% of `train` (stratified, seed 42) is held out as the validation set and is used to pick the best epoch. The `test` folder is used only once, for the final numbers below.
- **Imbalance:** disgust has about 16x fewer images than happy. Training draws batches with a weighted sampler, weights proportional to 1/sqrt(class count), so rare classes are seen more often without heavy oversampling.
- **Preprocessing:** images are converted to grayscale, resized to the model's input size (48 for the CNN, 96 for ResNet-18) and normalised to [-1, 1].
- **Augmentation** (training only): horizontal flip, rotation up to ±15°, scaling 0.85–1.15, translation up to 10%, brightness/contrast jitter, and random erasing.

## 3. Models
**DeepFERNet (custom CNN, 4.8M parameters).** Four blocks, each made of two 3x3
conv-BatchNorm-ReLU layers, max-pooling and dropout (0.1 to 0.3), with 64, 128, 256 and 512
channels. Global average pooling feeds a 256-unit dense layer (dropout 0.5) and a 7-way
output. Trained for 30 epochs.

**ResNet-18 transfer learning (11.2M parameters).** ImageNet-pretrained ResNet-18. The gray
channel is copied to RGB, and the final layer is replaced by dropout plus a 7-way linear
layer. Training has two stages. For 2 epochs only the new head is trained, at learning rate
1e-3. Then all layers are fine-tuned for 13 epochs at learning rate 2e-4.

**Shared training setup.** AdamW optimiser (weight decay 1e-4), one-cycle learning-rate
schedule, cross-entropy with label smoothing 0.1, batch size 128. The checkpoint with the
best validation accuracy is kept. Training ran on an Apple M5 GPU (MPS), at about 50 s per
epoch for the CNN and 60 s for ResNet-18.

## 4. Results (held-out test set, 7,178 images)

| model | accuracy | macro precision | macro recall | macro F1 | latency / face* |
|---|---:|---:|---:|---:|---:|
| DeepFERNet (CNN) | 67.0% | 63.3% | 64.9% | 63.9% | 0.73 ms |
| ResNet-18 (transfer) | 67.6% | 66.9% | 65.0% | 65.5% | 1.62 ms |
| Ensemble (average of both) | **69.3%** | **67.3%** | **67.3%** | **67.0%** | ~2.4 ms |

\*Batch size 1, Apple M5 GPU, model only. Face detection is not included.

For context, human accuracy on FER2013 is estimated at about 65%, and published single
models usually reach 65–73%.

### Per-class F1 (ResNet-18)
| angry | disgust | fear | happy | neutral | sad | surprise |
|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.66 | 0.49 | 0.87 | 0.65 | 0.52 | 0.79 |

Confusion matrices are in `runs/cnn_confusion.png` and `runs/resnet18_confusion.png`, and
the training curves are in `runs/training_curves.png`.

## 5. Findings
- **Easiest and hardest classes.** Happy (88% recall) and surprise (83%) have distinctive
  mouth and eye shapes. Fear (43%) and sad (48%) are the hardest.
- **Confusions.** Fear is spread across sad, angry, neutral and surprise. Sad is most often
  predicted as neutral (21%). Disgust is mostly confused with angry (28%), which matches the
  facial muscles the two expressions share.
- **Effect of balancing.** Class-balanced sampling lifted disgust to 58–62% recall despite
  its tiny training set.
- **Transfer learning.** ResNet-18 converged faster (67% validation accuracy by epoch 11,
  against about 29 epochs for the CNN) and had better macro precision. It also began to
  overfit (train 75% against val 68%). The CNN still had a small gap at epoch 30 because of
  heavy dropout and augmentation, so longer training would probably help it.
- **Ensemble.** Averaging the two models' probabilities adds about 1.7 points of accuracy.
  This suggests their errors are partly independent.

## 6. Real-time system and application
- `emotion.py` detects faces with OpenCV's Haar cascade, crops and resizes each one, and
  classifies all faces in a single batch.
- `realtime.py` runs on a live webcam feed and draws a box, label and confidence for each
  face, plus an FPS counter. The model takes about 1.6 ms per face, so frame rate is limited
  by the camera and the face detector, not the network.
- `app.py` is a Streamlit app that takes a webcam snapshot or an uploaded image and shows
  the annotated faces with a probability bar chart. If no face is found it classifies the
  whole image, so pre-cropped dataset images also work.

## 7. Limitations and next steps
- FER2013 labels are noisy, and 48x48 images throw away detail, which caps accuracy at
  roughly 75%.
- The Haar detector misses profile and tilted faces. A DNN detector (OpenCV's SSD or
  MediaPipe) would be more reliable.
- Possible improvements: test-time augmentation (flip averaging), a larger backbone
  (ResNet-34 or EfficientNet) at higher resolution, MixUp/CutMix, and temporal smoothing of
  predictions across video frames.
- Not done: deployment with real users and collection of their feedback (objective 8), and
  live webcam testing on a real camera. The webcam path was tested on synthetic frames, and
  the app was checked to load and run.
