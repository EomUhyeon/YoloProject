import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../scooter_yolov5/runs/train/exp11/weights/best.pt')

# 검출 이미지 경로
img_path = '../test_img_001.jpg'

results = model(img_path)

# 결과 출력 및 시각화
results.show()  # 이미지에 바운딩 박스 그리기
results.save()  # 결과 저장
