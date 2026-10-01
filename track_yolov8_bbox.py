# from collections import defaultdict

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO("D:/COS_YOLOv8/240529_COS_bbox/240602_best.pt")

# # Open the video file
# video_path = './input_data/COS(7-1)P1_L2.MTS'
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True)
#         # defalt tracker는 botsort.yaml임, 바꾸고자한다면
#         # results = model.track(frame, tracker='botsort.yaml', persist=True)와 같이 tracker를 지정해야함
#         for result in results:
#             boxes = result.boxes  # Boxes object for bounding box outputs
#             masks = result.masks  # Masks object for segmentation masks outputs
#             keypoints = result.keypoints  # Keypoints object for pose outputs
#             probs = result.probs  # Probs object for classification outputs
#             obb = result.obb  # Oriented boxes object for OBB outputs

#             if not boxes.conf.nelement() == 0:
#                 for box in boxes:
#                     # bbox를 그리기 위한 정보(class 넘버, 확률값, xy좌표)
#                     # Get the boxes and track IDs
#                     box_cls_num = box.cls.tolist()
#                     box_conf = box.conf.tolist()
#                     box_data = box.xyxy.tolist()  
#                     box_data_xywh = box.xywh.tolist()
#                     track_ids = box.id.tolist()
#                     # # Get the boxes and track IDs
#                     # boxes = results[0].boxes.xywh.cpu()
#                     # track_ids = results[0].boxes.id.int().cpu().tolist()

#                     # Visualize the results on the frame
#                     annotated_frame = results[0].plot()
#                     # data를 꺼내려 한다면 draw box 함수로 변경하기

#                     # Plot the tracks
#                     for box, track_id in zip(box_data_xywh, track_ids):
#                         x, y, w, h = box
#                         track = track_history[int(track_id)]
#                         track.append((float(x), float(y)))  # x, y center point
#                         if len(track) > 30:  # retain 90 tracks for 90 frames
#                             track.pop(0)

#                         # Draw the tracking lines
#                         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                         cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
#             else:
#                 annotated_frame = frame
#             # Display the annotated frame
#             cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load the model with tracking enabled
# model = YOLO("yolov8n.pt", tracker="bytetrack.yaml")
model = YOLO(r"D:\COS_YOLOv8\240529_COS_bbox\240606_best.pt")

# Open the video file
video_path = './input_data/2.1_1set_pair2_L1_Trim.MTS'
output_path = './mod_video_trim_test_0607.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 속성 확인
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 동영상 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking on the current frame
    results = model.track(frame, tracker="botsort.yaml", persist=True)
    # results = model.track(frame, tracker="bytetrack.yaml", persist=True, save_txt=True)

    # Check if tracking IDs are available
    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Draw boxes and IDs on the frame
        for box, id in zip(boxes, ids):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Id {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("Tracking IDs not available.")

    # Display the frame
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    out.write(frame)

# Release resources
cap.release()
cv2.destroyAllWindows()

print("done")