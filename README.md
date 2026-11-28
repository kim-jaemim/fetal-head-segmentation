# Fetal Head Ultrasound Segmentation and HC Measurement

## 1. Project Introduction

HC18 태아 머리 초음파 데이터셋을 이용하여 태아 머리 영역을 분할하고 머리둘레(HC)를 측정하는 프로젝트입니다.

고전 영상처리 기법과 딥러닝 기반 영상 분할 모델의 성능을 비교합니다.

## 2. Dataset

- Dataset: HC18
- Download: https://zenodo.org/records/1327317
- Training images: 853
- Test images: 146

원본 데이터셋은 GitHub에 업로드하지 않으며, 제공된 링크에서 다운로드합니다.

## 3. Methods

본 프로젝트에서는 다음 세 가지 방법을 비교합니다.

1. Classical Image Processing
   - Canny Edge Detection
   - Ellipse Fitting

2. U-Net
   - U-Net 기반 태아 머리 영역 분할
   - ResNet34 Encoder 적용
   - Attention U-Net 추가 구현
   - Attention Gate를 활용한 특징 영역 집중

3. FCN
   - FCN 기반 태아 머리 영역 분할

## 4. Data Split

데이터셋은 환자 단위로 분할하여 데이터 누수를 방지합니다.

- Train / Validation: 5-Fold 교차검증
- Test: 최종 성능 평가용으로 별도 보관

각 Fold는 학습 데이터와 검증 데이터로 구성하며,
5개 Fold의 검증 결과를 종합하여 평균 성능을 계산합니다.

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
│   └── folds.csv
├── src/
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── augment.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   ├── classical.py
│   ├── unet.py
│   └── fcn.py
├── notebooks/
└── minutes/
```
