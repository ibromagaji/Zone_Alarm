Zone Alarm 

Real-time object detection for security cameras — get an alarm and a Telegram photo the moment something shows up.

Runs on-device with a quantized YOLO11n model, so it stays fast even on older, low-power hardware. No cloud inference, no subscription — just a webcam/IP camera and a laptop.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TFLite](https://img.shields.io/badge/model-YOLO11n%20INT8%20TFLite-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

How it works

A YOLO11n model is exported and INT8-quantized to TFLite with a 320×320 input, trading a bit of accuracy for speed. This keeps the pipeline light enough to run in real time on modest CPUs — around 14 FPS on a Lenovo ThinkPad T430 (no GPU).

When an object is detected:
1. A local alarm sound is triggered.
2. The detection frame (with bounding box) is sent to a Telegram chat via bot API.

Two modes, two branches

This repo ships two behaviors as separate branches:

| Branch | Behavior |
|---|---|
| [`alarm`](https://github.com/ibromagaji/Zone_Alarm/tree/alarm) | Detects objects anywhere in frame. Any detection fires the alarm + Telegram alert. |
| [`overlay.py`](https://github.com/ibromagaji/Zone_Alarm/tree/overlay.py) | Zone-based detection. Click 4 points on the video feed to draw a custom polygon zone — only detections inside that zone trigger the alarm. |


Features

- Lightweight YOLO11n, INT8-quantized TFLite for fast CPU inference
- Optional custom detection zone — click 4 points, only that region matters
- Instant local alarm on detection
- Sends the annotated detection frame straight to Telegram
- Bring your own model — just drop in your `.tflite` file

Getting started

Requirements
- Python 3.9+
- A webcam or IP/RTSP camera feed
- A `.tflite` model (YOLO11n INT8, 320×320 input) — export your own or use a compatible one
- A Telegram bot token + chat ID ([BotFather](https://t.me/BotFather) setup)

Installation

```bash
git clone https://github.com/ibromagaji/Zone_Alarm.git
cd Zone_Alarm

# choose your mode
git checkout alarm          # any-object, anywhere detection
# or
git checkout overlay.py     # zone-based detection

pip install -r requirements.txt
```

Add your model

Drop your quantized `.tflite` model into the project directory (or update the model path in the script) — the project is plug-and-play with any compatible YOLO11n TFLite export.

Configure Telegram

Bot token and chat ID are currently hardcoded in the script. Open the main script and set:

```python
BOT_TOKEN = "your-telegram-bot-token"
CHAT_ID = "your-chat-id"
```

TODO: Move these to a `.env` file (e.g. via `python-dotenv`) instead of hardcoding, so credentials aren't committed to the repo.

Run it

```bash
python main.py
```

On the `overlay.py` branch, a window will open on first run — click 4 points on the frame to define your detection zone before it starts monitoring.

Performance

| Model | Input size | Quantization | FPS (CPU only) |
|---|---|---|---|
| YOLO11n | 320×320 | INT8 (TFLite) | ~14 FPS on ThinkPad T430 |

Roadmap / ideas

- [ ] Move Telegram credentials to `.env`
- [ ] Support multiple zones per feed
- [ ] Configurable cooldown between alerts (avoid spam on repeated detections)
- [ ] Optional recording of a short clip instead of a single frame

License

MIT — feel free to use, modify, and build on this.