# Fetal Head Ultrasound Segmentation and HC Measurement

## 1. Project Introduction

HC18 태아 머리 초음파 데이터셋을 이용하여 태아 머리 영역을 분할하고 머리둘레(HC)를 측정하는 프로젝트입니다.

고전 영상처리 기법과 딥러닝 기반 영상 분할 모델의 성능을 비교합니다.

## 2. Dataset

- Dataset: HC18
- Download: https://zenodo.org/records/1327317
- Training images: 999

원본 데이터셋은 GitHub에 업로드하지 않으며, 제공된 링크에서 다운로드합니다.

## 3. Methods

본 프로젝트에서는 다음 세 가지 방법을 비교합니다.

1. Classical Image Processing
   - Canny Edge Detection
   - Ellipse Fitting

2. U-Net
   - U-Net 기반 태아 머리 영역 분할

3. FCN
   - FCN 기반 태아 머리 영역 분할

## 4. Data Split

데이터셋은 다음과 같이 분할할 계획입니다.

- Train: 70%
- Validation: 15%
- Test: 15%

데이터 분할 목록은 `splits/` 폴더의 CSV 파일로 관리합니다.

## 5. Evaluation

각 방법의 성능을 다음 기준으로 비교합니다.

- IoU
- Dice Score
- Processing Time
- Model Parameters
- HC Measurement Error

## 6. Development Environment

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Google Colab

## 7. Project Structure

```text
fetal-head-segmentation/
├── README.md
├── .gitignore
├── requirements.txt
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── classical.py
│   ├── unet.py
│   └── fcn.py
├── notebooks/
└── minutes/
```
