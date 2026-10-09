# Test Images for Facial Emotion Recognition

This directory contains curated, high-resolution test images designed to validate both single-face and multi-face detection as well as emotion classification across primary emotion categories.

---

## 📁 Included Test Images

| Filename | Expected Emotion / Type | Description |
| :--- | :--- | :--- |
| `happy_test.jpg` | **Happy** | High-definition portrait with clear joyful smiling expression (100% confidence). |
| `sad_test.jpg` | **Sad** | High-definition portrait with melancholic, sorrowful expression (66.8% confidence). |
| `angry_test.jpg` | **Angry** | High-definition portrait with furrowed brow and anger expression (78.8% confidence). |
| `surprise_test.jpg` | **Surprise** | High-definition portrait with wide eyes and open-mouth shock (98.9% confidence). |
| `neutral_test.jpg` | **Neutral** | High-definition portrait with relaxed, baseline calm expression (79.4% confidence). |
| `fear_test.jpg` | **Fear** | Portrait displaying startled, frightened facial markers (51.8% confidence). |
| `disgust_test.jpg` | **Disgust / Angry** | Portrait with wrinkled nose and strong revulsion cues. |
| `multiface_test.jpg` | **Multi-Face Crowd** | Complex multi-person image to test OpenCV Haar cascade multi-face extraction (7 faces detected). |

---

## 🚀 Running Automated Batch Tests

You can run the automated evaluation script directly from this directory or root:

```bash
# Run batch test script
python test_images/run_tests.py
```

### Or Test an Individual Image with the CLI:

```bash
# Test single image
python cli.py face --image test_images/happy_test.jpg

# Save annotated detection with bounding boxes
python cli.py face --image test_images/multiface_test.jpg --output test_images/multiface_detected.jpg
```

---

## 🌐 Web & API Testing

These images are also pre-copied to:
- `assets/samples/faces/` (Available to the FastAPI backend at `http://localhost:8000/api/samples`)
- `frontend/public/media/samples/faces/` (Available to the React frontend on Vercel)
