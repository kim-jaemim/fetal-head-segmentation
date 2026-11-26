# Canny 엣지 검출 및 타원 피팅을 이용한 고전 영상처리

import cv2
import numpy as np


def detect_edges(img, threshold1=30, threshold2=100):
    edges = cv2.Canny(
        img,
        threshold1,
        threshold2
    )
    return edges


def apply_morphology(edges):
    """
    Morphology 연산으로 끊어진 경계 연결
    """

    kernel = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed


def find_contours(edges):
    """
    이미지에서 외곽선 추출
    """

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def select_head_contour(contours):
    """
    태아 머리 윤곽선 후보 선택
    """

    candidates = []

    for contour in contours:

        # 타원 피팅에는 최소 5개의 점이 필요
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)

        # 너무 작은 윤곽선은 제외
        if area < 1000:
            continue

        candidates.append(contour)

    # 후보가 없으면 None 반환
    if not candidates:
        return None

    # 면적이 가장 큰 윤곽선 선택
    best_contour = max(
        candidates,
        key=cv2.contourArea
    )

    return best_contour


def fit_head_ellipse(contour):
    """
    태아 머리 윤곽선에 타원 피팅
    """

    if contour is None or len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)

    return ellipse


def calculate_hc(ellipse):
    """
    타원의 둘레 계산 (픽셀 단위)
    Ramanujan 근사 공식 사용
    """

    if ellipse is None:
        return None

    # 타원의 가로, 세로 지름
    width, height = ellipse[1]

    # 반장축과 반단축
    a = width / 2
    b = height / 2

    # Ramanujan 타원 둘레 근사 공식
    hc = np.pi * (
        3 * (a + b)
        - np.sqrt((3 * a + b) * (a + 3 * b))
    )

    return hc



def predict(img):
    """
    초음파 이미지에서 태아 머리 영역을 검출하고
    0/1 마스크를 반환
    """

    # 1. Canny 엣지 검출
    edges = detect_edges(img)

    # 2. Morphology 연산
    processed = apply_morphology(edges)

    # 3. 윤곽선 추출
    contours = find_contours(processed)

    # 4. 태아 머리 윤곽선 선택
    contour = select_head_contour(contours)

    # 5. 타원 피팅
    ellipse = fit_head_ellipse(contour)

    # 6. 타원 마스크 생성
    mask = create_ellipse_mask(img.shape, ellipse)

    return mask



def create_ellipse_mask(img_shape, ellipse):
    """
    타원 정보를 이용해 0/1 마스크 생성
    """

    if ellipse is None:
        return np.zeros(img_shape, dtype=np.uint8)

    mask = np.zeros(img_shape, dtype=np.uint8)

    cv2.ellipse(
        mask,
        ellipse,
        1,
        thickness=-1
    )

    return mask
