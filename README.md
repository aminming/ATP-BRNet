# ATP-BRNet

This repository contains the codebase for **ATP-BRNet**, a two-phase deep learning framework for predicting ATP levels in organoid bright-field images.

## 📦 Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Project Structure

### Phase 1: Classification Module Training

Train the ATP level classification model:

```bash
python AD2DMIT_train_logbin_concat_3.py
```

---

### Phase 2: Regression Module Training

Train the multi-head regression model with bin-based supervision:

```bash
python ADMutiHead_train.py
```

---

### 🔍 Inference on JSON File

Make predictions from a given image-ATP mapping JSON:

```bash
python pred_csv_MutiQ.py
```

---

### 🌡️ Heatmap Generation

Generate and save predicted ATP heatmaps:

```bash
python pred_csv_MutiQ_save.py
```

---

## 📖 Citation

If you find **ATP-BRNet** useful in your research, please consider citing our work:

```bibtex
# Citation placeholder
```

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact the authors.
