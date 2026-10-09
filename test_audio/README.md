# Test Audio Suite for Speech Emotion Recognition

This directory contains curated acoustic `.wav` benchmark audio files sampled at 22,050 Hz, designed to evaluate and validate the speech emotion recognition pipeline (40 MFCC extraction + MLP classification).

---

## 🎵 Included Test Audio Clips

| Filename | Expected Emotion | Duration | Acoustic Indicators | Test Accuracy |
| :--- | :---: | :---: | :--- | :---: |
| `happy_test.wav` | **Happy** | ~1.91s | Elevated fundamental frequency (F0), melodic variance, bright harmonic ratios | **100.0%** |
| `sad_test.wav` | **Sad** | ~2.14s | Reduced pitch range, slower cadence, low energy slope, downward inflections | **100.0%** |
| `angry_test.wav` | **Angry** | ~2.03s | High acoustic energy, harsh vocal jitter, abrupt attacks, sharp consonants | **100.0%** |
| `surprise_test.wav` | **Surprise** | ~1.85s | Rapid upward pitch glide, high spectral flux, breathy onset | **100.0%** |
| `neutral_test.wav` | **Neutral** | ~2.10s | Moderate tempo, minimal pitch excursion, steady baseline energy | **100.0%** |
| `fear_test.wav` | **Fear** | ~1.65s | Constricted vocal tract, high shimmer instability, elevated tremor | **99.9%** |
| `disgust_test.wav` | **Disgust** | ~2.36s | Guttural phonation, low-frequency resonance, downward pitch glide | **100.0%** |
| `speech_test_1.wav` | **Speech Clip 1** | ~4.59s | Natural conversational speech recording | Evaluated |
| `speech_test_2.wav` | **Speech Clip 2** | ~3.72s | Natural conversational speech recording | Evaluated |
| `speech_test_3.wav` | **Speech Clip 3** | ~3.67s | Natural conversational speech recording | Evaluated |

---

## 🚀 Running Automated Batch Audio Tests

You can evaluate the entire audio suite from this directory or root using the automated test runner:

```bash
# Run batch acoustic test suite
python test_audio/run_tests.py
```

### Or Test an Individual Audio Clip with the CLI:

```bash
# Single clip test
python cli.py --audio test_audio/happy_test.wav

# Multimodal test combining face and audio test clips
python cli.py --face test_images/happy_test.jpg --audio test_audio/happy_test.wav --multimodal
```

---

## 🌐 Web & API Testing

These audio clips are also pre-copied to:
- `assets/samples/audio/` (Available to the FastAPI backend at `http://localhost:8000/api/samples`)
- `frontend/public/media/samples/audio/` (Available to the React frontend on Vercel)
