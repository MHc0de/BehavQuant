from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def draw_boxes(img, boxes, cls, conf, names):
  x1 = int(boxes[0])
  y1 = int(boxes[1])
  x2 = int(boxes[2])
  y2 = int(boxes[3])

#   # 랜덤 색상 생성
#   box_color = [random.randint(0, 255) for _ in range(3)]
  cls_num = cls[0]
  cls_name = names.get(int(cls_num))
  # print(cls_name)

  if cls_num == 0.0: #rat
      box_color = (0, 0, 255)
  elif cls_num == 1.0: #toy
      box_color = (0, 255, 0)
  else:
      box_color = (255, 0, 0)

  label = f"{cls_name}: {conf[0]:.2f}"

  # bbox 그리기
  # for x1, y1, x2, y2 in boxes:
  cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness=2)

  # For the text background
  # Finds space required by the text so that we can put a background with that amount of width.
  (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
  text_box_coord1 = (x1, y1 - 20) #text를 위에 씀
  text_box_coord2 = (x1 + w, y1)
  text_coord = (x1, y1 - 5)

  # Prints the text.
  img = cv2.rectangle(img, text_box_coord1, text_box_coord2, box_color, -1)
  img = cv2.putText(img, label, text_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

  return img

model = YOLO("D:/COS_YOLOv8/240529_COS_bbox/240602_best.pt")

video_path = './input_data/COS(7-1)P1_L2.MTS'
output_path = './mod_video.mp4'

cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture('./input_data/C L1(F).MTS_1240.jpg')

# 비디오 속성 확인
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 동영상 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    names = results[0].names

    # annotated_frame = results[0].plot()
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        # print(boxes.conf)
        if not boxes.conf.nelement() == 0:
            for box in boxes:
                # bbox를 그리기 위한 정보(class 넘버, 확률값, xy좌표)
                box_cls_num = box.cls.tolist()
                box_conf = box.conf.tolist()
                box_data = box.xyxy.tolist()  
                # print(box_cls_num)    
                annotated_frame_with_data = draw_boxes(frame, box_data[0], box_cls_num, box_conf, names)


            # 결과 비디오에 프레임 쓰기
            out.write(frame)

            # cv2.imshow("test", frame) ## 이미지 인터프린터에 출력

            # # 'q'를 누르면 루프 종료
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # # 아무 키나 누를 때까지 유지
            # cv2.waitKey(0)
            # # 모든 윈도우 닫기
            # cv2.destroyAllWindows()

# 자원 해제
cap.release()
cv2.destroyAllWindows()

print("done")