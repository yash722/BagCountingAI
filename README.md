# Real-Time Bag Counter Using YOLOv8 + Deep SORT

This project implements a real-time object counting system using [YOLOv8](https://github.com/ultralytics/ultralytics) and [Deep SORT](https://github.com/levan92/deep_sort_realtime), specifically designed for CCTV footage where a camera is fixed toward a gate or entry zone.

The system counts objects (like bags or people) that enter and exit a **defined rectangular zone**. Counts are updated based on the **direction of movement** (entering from top or bottom).

---

## ✅ Features

- Object detection using **YOLOv8**
- Multi-object tracking using **Deep SORT**
- Rectangular zone-based **entry/exit counter**
- Direction-based counting (up ↗ increases, down ↘ decreases)

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yash722/BagCountingAI.git
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run main.py

```bash
python main.py
```
