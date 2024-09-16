import sys
import os

# yolov5 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))  # yolov5 경로 추가

import torch
import cv2  # OpenCV for visualization
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

# 커스텀 YOLOv5 모델 경로 지정 (사용자 커스텀 모델 경로로 수정)
custom_model_path = 'helmet_yolov5/helmet_yolov5_001.pt'  # 커스텀 모델 가중치 경로

# 로컬에서 커스텀 YOLOv5 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(custom_model_path, device=device)  # 커스텀 모델 가중치 파일 로드
model.eval()  # 평가 모드로 설정

# 이미지 경로를 통해 예측 수행
img_path = 'test_img/test_cctv_012.png'

# OpenCV를 사용해 이미지를 로드
im0s = cv2.imread(img_path)

# 이미지가 정상적으로 불러와졌는지 확인
if im0s is None:
    print(f"Failed to load image at {img_path}")
    sys.exit()

# 이미지 전처리 (YOLOv5가 사용하는 입력 크기와 동일하게 조정)
img = cv2.resize(im0s, (640, 640))  # YOLOv5의 입력 크기
img = img[:, :, ::-1].copy()  # BGR to RGB 변환 후 copy()로 negative stride 문제 해결
img = img.transpose(2, 0, 1)  # HWC to CHW 변환
img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)  # 정규화 및 텐서 변환

# 모델 예측
with torch.no_grad():
    pred = model(img)

# Non-Max Suppression (NMS) 적용하여 중복된 바운딩 박스 제거
pred = non_max_suppression(pred, 0.25, 0.45)  # conf_threshold=0.25, iou_threshold=0.45

# 클래스 이름 정의 (커스텀 모델에 맞는 클래스 레이블을 사용)
class_names = model.names

# 시각화 함수 정의
def plot_boxes_on_image(im0s, det, class_names):
    for *xyxy, conf, cls in det:
        # 바운딩 박스 좌표 변환 (이미 원본 이미지 크기와 동일하므로 변환 불필요)
        label = f'{class_names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, xyxy)  # 좌표를 정수로 변환
        # 바운딩 박스 그리기
        cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 클래스 레이블 및 신뢰도 표시
        cv2.putText(im0s, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 예측 결과 시각화
for det in pred:
    if len(det):
        # 바운딩 박스 및 클래스 시각화
        plot_boxes_on_image(im0s, det, class_names)

        # 결과 이미지 출력
        cv2.imshow('Detection', im0s)
        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()  # 창 닫기

        # 결과 이미지 저장
        cv2.imwrite('result_image.jpg', im0s)  # 결과 저장
