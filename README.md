# DeepFER: Facial Emotion Recognition Using Deep Learning

Implements the DeepFER project brief (Deep Learning for Computer Vision) on the supplied
FER2013 dataset: 7 emotions, 48x48 grayscale faces, 28,709 train and 7,178 test images.
See `REPORT.md` for the method and results.

**Test accuracy:** CNN 67.0% · ResNet-18 transfer 67.6% · ensemble 69.3%

| Brief objective | Where |
|---|---|
| 1. Preprocessing + augmentation (rotation, scaling, flipping) | `common.py` `train_transform` |
| 2. Custom CNN + transfer learning (fine-tuned pre-trained model) | `common.py` `DeepFERNet`, `ResNetTL` |
| 3. Training, accuracy / precision / recall / F1 | `train.py`, `evaluate.py` |
| 4. Real-time processing from live video | `realtime.py`, `emotion.py` |
| 5. User-friendly application | `app.py` (Streamlit) |
| 6. Performance / latency | `evaluate.py` (ms per image), `realtime.py` (FPS) |
| 7. Documentation and report | `REPORT.md`, `runs/*.png` |

Trained models are included in `runs/` (`cnn.pt`, `resnet18.pt`), so you can skip training.

## Setup (once)
```bash
cd ~/Downloads/deepfer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
In every new Terminal window, run `cd ~/Downloads/deepfer && source venv/bin/activate` first.

## Use the trained models
```bash
python realtime.py                    # webcam, press q to quit (macOS: allow camera for Terminal)
python realtime.py --ckpt runs/cnn.pt # same, with the custom CNN
streamlit run app.py                  # web app: webcam snapshot or image upload
python evaluate.py --data ~/Downloads/archive-3   # re-create metrics and plots in runs/
```

## Retrain (optional, ~25 + ~15 min on Apple Silicon)
```bash
python train.py --data ~/Downloads/archive-3 --model cnn --epochs 30
python train.py --data ~/Downloads/archive-3 --model resnet18 --epochs 15 --freeze-epochs 2
```
`--data` is the folder that contains `train/` and `test/`. The ResNet-18 ImageNet weights
are downloaded by torchvision on first use.
