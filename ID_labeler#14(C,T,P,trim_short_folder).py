from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm  # 진행률 바를 표시하기 위한 라이브러리
import shutil
import cv2
import numpy as np
import csv
import os

def hyotrack_csv_save_table(save_path, table_file_name, header, frame_num, data):
    table_save_path = os.path.join(save_path, table_file_name)
    
    data.insert(0, frame_num)
    table = data.copy()

    file_exists = os.path.exists(table_save_path)
    
    with open(table_save_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(header)
        csv_writer.writerow(table)

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")

def delete_folder_contents(folder_path):
    try:
        # 폴더가 실제로 존재하는지 확인합니다
        if os.path.isdir(folder_path):
            # 폴더의 모든 항목을 반복하며 삭제합니다
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    # 폴더인 경우, 하위 폴더 및 파일을 재귀적으로 삭제합니다
                    shutil.rmtree(item_path)
                else:
                    # 파일인 경우, 직접 삭제합니다
                    os.remove(item_path)
            print(f"All contents of {folder_path} have been deleted successfully.")
        else:
            print(f"{folder_path} is not a directory.")
    except Exception as e:
        print(f"An error occurred while trying to delete the contents: {e}")

def make_folder(data_path, save_folder_name):
    # 경로 상에 폴더가 존재하는지 확인 후 없다면 폴더 생성
    save_path = os.path.join(data_path, save_folder_name)
    if not os.path.exists(save_path): # 경로가 존재하지 않으면
        os.makedirs(save_path, exist_ok=True) # 경로 생성 (mkdir은 마지막 경로만 만들음, makedirs는 경로가 없으면 상위까지 전부 만들어줌)
    else:
        print(f"경로 '{save_path}'이(가) 이미 존재합니다.")
    return save_path

def save_fail_list(fail_list, save_path):
    
    faillist_save_path = os.path.join(save_path, "fail_list.txt")
    delete_file(faillist_save_path) # 처음 실행시 기존 파일 삭제

    if not os.path.exists(faillist_save_path):
        with open('fail_list.txt', 'w') as txtfile:
            for video_path, error_message in fail_list:
                txtfile.write(f"{video_path}: {error_message}\n")
    else:
        with open('fail_list.txt', 'a') as txtfile:  # 'a' 모드로 파일 열기 (추가 모드)
            for video_path, error_message in fail_list:
                txtfile.write(f"{video_path}: {error_message}\n")

def read_csv_with_header(file_path):
    # CSV 파일을 읽어 첫 줄은 헤더로, 나머지 줄은 데이터로 반환하는 함수
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data_list = [row for row in csvreader]
    return header, data_list

def find_rows_with_value(data_list, target_value, col_index):
    # 데이터 리스트에서 특정 열의 특정 값이 일치하는 행 번호를 반환하는 함수
    results = []
    for idx, row in enumerate(data_list):
        if row[col_index] == target_value:
            results.append(idx)
    return results

def get_xyxy(data_row, columns):
    return [float(data_row[col]) for col in columns]

def set_xyxy(data_row, columns, xyxy_values):
    for i, col in enumerate(columns):
        data_row[col] = xyxy_values[i]

def write_csv_with_header(file_path, csv_file_name, header, data):
    # 보간을 완료한 CSV 파일을 저장하는 함수

    interpolation_save_path = os.path.join(file_path, csv_file_name)

    with open(interpolation_save_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(data)

def process_results(results, data, interpolate_column_start, interpolate_column_end, cls_num):
    n = len(results)
    i = 0
    start = 0
    end = 0
    start_num = 0
    end_num = 0
    while i < n:
        start = i
        end = i
        while i < n - 1 and results[i] + 1 == results[i + 1]:
            i += 1
            end = i   
        start_num = results[start]
        end_num = results[end]

        if start_num == 0:
            prev_data = post_data = data[end_num + 1]            
            prev_xyxy = post_xyxy = np.array(post_data[interpolate_column_start:interpolate_column_end+1], dtype=int)
        elif end_num == int(data[-1][0]):
            prev_data = post_data = data[start_num - 1]            
            prev_xyxy = post_xyxy = np.array(prev_data[interpolate_column_start:interpolate_column_end+1], dtype=int)
        else:
            prev_data = data[start_num - 1]
            post_data = data[end_num + 1]
            prev_xyxy = np.array(prev_data[interpolate_column_start:interpolate_column_end+1], dtype=int)
            post_xyxy = np.array(post_data[interpolate_column_start:interpolate_column_end+1], dtype=int)

        if start == end:  # 연속된 숫자가 아닌 경우
            # print(start_num)
            # print(end_num)
            if start >= 0 and end <= len(results) - 1:
                interpolated_xyxy = (prev_xyxy + post_xyxy) / 2
                data[start_num][interpolate_column_start-1] = cls_num
                data[start_num][interpolate_column_start:interpolate_column_end+1] = interpolated_xyxy.astype(int).tolist()                
        else:  # 연속된 숫자인 경우
            if post_data[1] == 'r':
                pass
            else:
                # print(start_num)
                # print(end_num)
                if start >= 0 and end <= len(results) - 1:
                    num_frames = end - start + 1
                    for j in range(1, num_frames + 1):
                        t = j / (num_frames + 1)
                        interpolated_xyxy = (1 - t) * prev_xyxy + t * post_xyxy
                        data[start_num + j - 1][interpolate_column_start-1] = cls_num
                        data[start_num + j - 1][interpolate_column_start:interpolate_column_end+1] = interpolated_xyxy.astype(int).tolist()
        i += 1
    return data

def calculate_iou(box1, box2):
    # IOU 계산 함수 정의

    box1_x1 = int(box1[0])
    box1_y1 = int(box1[1])
    box1_x2 = int(box1[2])
    box1_y2 = int(box1[3])

    box2_x1 = int(box2[0])
    box2_y1 = int(box2[1])
    box2_x2 = int(box2[2])
    box2_y2 = int(box2[3])

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou
    
def newframe_or_not(new_frame, box_info, cls_num, x1, y1, x2, y2, id, conf):
    if new_frame == None: # 이전 프레임이 없을때
        box_info = [cls_num, x1, y1, x2, y2, id, conf]
        new_frame = False
    elif new_frame == True: # 이전 프레임과 다른 새로운 프레임일때
        box_info = [cls_num, x1, y1, x2, y2, id, conf]
        new_frame = False
    elif new_frame == False: # 이전프레임과 같은 기존 프레임일때 
        if conf > box_info[6]:
            box_info = [cls_num, x1, y1, x2, y2, id, conf]
        else:
            pass
    else:
        print("new frame key is not exist")
    return box_info

def one_rat(new_frame, tmp_save_folder, csv_file_name, header, frame_num, save_folder):
    # best 값 초기화
    best_box_info = ["r", 0, 0, 0, 0, 0, 0]
    best_box_info_tmp = ["r", 0, 0, 0, 0, 0, 0]

    # tracking에 사용할 training 모델의 weight 파일을 Load
    model = YOLO(model_path)

    # video frame을 분석하는 과정 (video 불러와서 frame 순서로 open)
    # 무한루프
    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        
        # 비디오 읽기 성공여부 확인
        if not ret:
            break # 비디오가 정상적으로 읽히지 않으면 종료

        # 현재 프레임에 대한 tracking 수행
        results = model.track(frame, tracker="botsort.yaml", persist=True) # botsort를 이용한 tracking
        # results = model.track(frame, tracker="bytetrack.yaml", persist=True, save_txt=True) # bytetrack을 이용한 tracking                               

        # tracking ID의 존재 여부 확인
        if results and results[0].boxes and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None: # ID가 있고 result가 있으면 실행
            
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            box_cls_num = results[0].boxes.cls.cpu().numpy().astype(int)
            box_conf = results[0].boxes.conf.tolist()

            # best 값 초기화
            best_box_info = ["r", 0, 0, 0, 0, 0, 0]
            
            for box, id, cls_num, conf in zip(boxes, ids, box_cls_num, box_conf):
                frame_num = int(frame_num)
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                id = int(id)
                cls_num = int(cls_num)
                conf = round(conf, 5)
                
                best_box_info_tmp = newframe_or_not(new_frame, best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                best_box_info = best_box_info_tmp.copy()

            # best값을 csv에 작성
            data = best_box_info
            hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)
            
            frame_num = frame_num + 1
            new_frame = True

        else: # ID가 없으면 문구 출력하기
            print("Tracking IDs not available.")

            best_box_info = ["r", 0, 0, 0, 0, 0, 0]
            data = best_box_info
            hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)                                        
        
            frame_num = frame_num + 1
            new_frame = True

    # 보간법 수행
    tmp_file_path = os.path.join(tmp_save_folder, csv_file_name) # CSV 파일 경로

    writed_header, interpolation_data = read_csv_with_header(tmp_file_path)
    rat_results = find_rows_with_value(interpolation_data, 'r', 1) # tmp에 저장한 data, 찾고자 하는 값, 검사할 열의 인덱스 (0 기반) 순서

    # 보간 처리
    rat_complete_data = process_results(rat_results, interpolation_data, 2, 5, 0)

    # CSV 파일에 헤더와 데이터를 덮어쓰기
    write_csv_with_header(save_folder, csv_file_name, writed_header, rat_complete_data)

    # results 초기화
    results.clear()

def one_rat_one_toy(new_frame, tmp_save_folder, csv_file_name, header, frame_num, save_folder):
    # best 값 초기화
    rat_best_box_info = ["r", 0, 0, 0, 0, 0, 0]
    rat_best_box_info_tmp = ["r", 0, 0, 0, 0, 0, 0]
    toy_best_box_info = ["t", 0, 0, 0, 0, 0, 0]
    toy_best_box_info_tmp = ["t", 0, 0, 0, 0, 0, 0]

    # tracking에 사용할 training 모델의 weight 파일을 Load
    model = YOLO(model_path)

    # video frame을 분석하는 과정 (video 불러와서 frame 순서로 open)
    # 무한루프
    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        
        # 비디오 읽기 성공여부 확인
        if not ret:
            break # 비디오가 정상적으로 읽히지 않으면 종료

        # 현재 프레임에 대한 tracking 수행
        results = model.track(frame, tracker="botsort.yaml", persist=True) # botsort를 이용한 tracking
        # results = model.track(frame, tracker="bytetrack.yaml", persist=True, save_txt=True) # bytetrack을 이용한 tracking                               

        # tracking ID의 존재 여부 확인
        if results and results[0].boxes and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None: # ID가 있고 result가 있으면 실행
            
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            box_cls_num = results[0].boxes.cls.cpu().numpy().astype(int)
            box_conf = results[0].boxes.conf.tolist()

            # best 값 초기화
            rat_best_box_info = ["r", 0, 0, 0, 0, 0, 0]
            toy_best_box_info = ["t", 0, 0, 0, 0, 0, 0]
            
            for box, id, cls_num, conf in zip(boxes, ids, box_cls_num, box_conf):
                frame_num = int(frame_num)
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                id = int(id)
                cls_num = int(cls_num)
                conf = round(conf, 5)
                if cls_num == 0: # rat
                    rat_best_box_info_tmp = newframe_or_not(new_frame, rat_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                    rat_best_box_info = rat_best_box_info_tmp.copy()

                elif cls_num == 1: #toy     
                    toy_best_box_info_tmp = newframe_or_not(new_frame, toy_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                    toy_best_box_info = toy_best_box_info_tmp.copy() 

            # best값을 csv에 작성
            data = rat_best_box_info + toy_best_box_info
            hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)
            
            frame_num = frame_num + 1
            new_frame = True

        else: # ID가 없으면 문구 출력하기
            print("Tracking IDs not available.")

            rat_best_box_info = ["r", 0, 0, 0, 0, 0, 0]
            toy_best_box_info = ["t", 0, 0, 0, 0, 0, 0]
            data = rat_best_box_info + toy_best_box_info
            hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)                                        
        
            frame_num = frame_num + 1
            new_frame = True

    # 보간법 수행
    tmp_file_path = os.path.join(tmp_save_folder, csv_file_name) # CSV 파일 경로

    writed_header, interpolation_data = read_csv_with_header(tmp_file_path)
    rat_results = find_rows_with_value(interpolation_data, 'r', 1) # tmp에 저장한 data, 찾고자 하는 값, 검사할 열의 인덱스 (0 기반) 순서
    toy_results = find_rows_with_value(interpolation_data, 't', 8)

    # 보간 처리
    rat_complete_data = process_results(rat_results, interpolation_data, 2, 5, 0)
    rat_toy_complete_data = process_results(toy_results, rat_complete_data, 9, 12, 1)

    # CSV 파일에 헤더와 데이터를 덮어쓰기
    write_csv_with_header(save_folder, csv_file_name, writed_header, rat_toy_complete_data)

    # results 초기화
    results.clear()

def two_rats(new_frame, width, tmp_save_folder, csv_file_name, header, frame_num, save_folder):
    # prev_id 초기화
    f_prev_id = 0
    p_prev_id = 0

    # best 값 초기화
    f_best_box_info = ["f", 0, 0, 0, 0, f_prev_id, 0]
    p_best_box_info = ["p", 0, 0, 0, 0, p_prev_id, 0]          

    # prev_iou 값 초기화
    prev_f_iou = 0.0
    prev_p_iou = 0.0

    # tracking에 사용할 training 모델의 weight 파일을 Load
    model = YOLO(model_path)

    # video frame을 분석하는 과정 (video 불러와서 frame 순서로 open)
    # 무한루프
    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()

        # prev_iou 값 초기화
        same_prev_f_iou = 0.0
        same_prev_p_iou = 0.0
        
        prev_f_iou = 0.0
        prev_p_iou = 0.0
        
        # 비디오 읽기 성공여부 확인
        if not ret:
            break # 비디오가 정상적으로 읽히지 않으면 종료

        # 현재 프레임에 대한 tracking 수행
        results = model.track(frame, tracker="botsort.yaml", persist=True) # botsort를 이용한 tracking
        # results = model.track(frame, tracker="bytetrack.yaml", persist=True, save_txt=True) # bytetrack을 이용한 tracking      
        if frame_num <= 30: 

            # center 기준
            center_x = width / 2 # half_width

            # tracking ID의 존재 여부 확인
            if results and results[0].boxes and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None: # ID가 있고 result가 있으면 실행
                
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                box_cls_num = results[0].boxes.cls.cpu().numpy().astype(int)
                box_conf = results[0].boxes.conf.tolist()

                # best 값 초기화
                f_best_box_info = ["f", 0, 0, 0, 0, f_prev_id, 0]
                f_best_box_info_tmp = ["f", 0, 0, 0, 0, f_prev_id, 0]
                p_best_box_info = ["p", 0, 0, 0, 0, p_prev_id, 0]
                p_best_box_info_tmp = ["p", 0, 0, 0, 0, p_prev_id, 0]
                
                for box, id, cls_num, conf in zip(boxes, ids, box_cls_num, box_conf):
                    frame_num = int(frame_num)
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    id = int(id)
                    cls_num = int(cls_num)
                    conf = round(conf, 5)
                    
                    if (x1+x2)/2 <= center_x: # 왼쪽

                        f_best_box_info_tmp = newframe_or_not(new_frame, f_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                        f_best_box_info = f_best_box_info_tmp.copy()

                        f_prev_id = f_best_box_info[5]

                    elif (x1+x2)/2 > center_x:
                        p_best_box_info_tmp = newframe_or_not(new_frame, p_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                        p_best_box_info = p_best_box_info_tmp.copy()
                        
                        p_prev_id = p_best_box_info[5]

                # best값을 csv에 작성
                data = f_best_box_info + p_best_box_info
                hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)
                
                frame_num = frame_num + 1
                new_frame = True

            else: # ID가 없으면 문구 출력하기
                print("Tracking IDs not available.")

                f_best_box_info = ["f", 0, 0, 0, 0, f_prev_id, 0]
                p_best_box_info = ["p", 0, 0, 0, 0, p_prev_id, 0]
                data = f_best_box_info + p_best_box_info
                hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)                                        
            
                frame_num = frame_num + 1
                new_frame = True
        else:
            if frame_num == 31:
                # 보간법 수행
                tmp_file_path = os.path.join(tmp_save_folder, csv_file_name) # CSV 파일 경로

                writed_header, interpolation_data = read_csv_with_header(tmp_file_path)

                f_results = find_rows_with_value(interpolation_data, 'f', 1) # tmp에 저장한 data, 찾고자 하는 값, 검사할 열의 인덱스 (0 기반) 순서
                p_results = find_rows_with_value(interpolation_data, 'p', 8)

                # 보간 처리
                f_complete_data = process_results(f_results, interpolation_data, 2, 5, 0)
                f_p_complete_data = process_results(p_results, f_complete_data, 9, 12, 0)

                # CSV 파일에 헤더와 데이터를 덮어쓰기
                write_csv_with_header(tmp_save_folder, csv_file_name, writed_header, f_p_complete_data)

                f_prev_id = f_p_complete_data[-1][6]
                p_prev_id = f_p_complete_data[-1][13]
                
                f_prev_box_info = f_p_complete_data[-1][1:8].copy()
                p_prev_box_info = f_p_complete_data[-1][8:15].copy()
            else:
                pass

            if results and results[0].boxes and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None: # ID가 있고 result가 있으면 실행
                
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                box_cls_num = results[0].boxes.cls.cpu().numpy().astype(int)
                box_conf = results[0].boxes.conf.tolist()
                
                for box, id, cls_num, conf in zip(boxes, ids, box_cls_num, box_conf):
                    frame_num = int(frame_num)
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    id = int(id)
                    cls_num = int(cls_num)
                    conf = round(conf, 5)

                    f_iou = 0
                    p_iou = 0

                    if f_prev_id in ids and p_prev_id in ids:
                        if f_prev_id == p_prev_id:
                            prev_f_xyxy = f_prev_box_info[1:5].copy()
                            prev_p_xyxy = p_prev_box_info[1:5].copy()
                            now_xyxy = [x1, y1, x2, y2]

                            f_iou = calculate_iou(prev_f_xyxy, now_xyxy)
                            p_iou = calculate_iou(prev_p_xyxy, now_xyxy)

                            if f_iou > p_iou:
                                if id == f_prev_id:
                                    f_best_box_info_tmp = newframe_or_not(new_frame, f_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                                    f_best_box_info = f_best_box_info_tmp.copy()

                                    f_prev_box_info = f_best_box_info.copy()
                                    f_prev_id = f_best_box_info[5]

                                else:
                                    # iou 검사해서 높으면 계속 p에 계속 갱신
                                    prev_p_xyxy = p_prev_box_info[1:5].copy()
                                    now_p_xyxy = [x1, y1, x2, y2]
                                    p_iou = calculate_iou(prev_p_xyxy, now_p_xyxy)

                                    if p_iou >= same_prev_p_iou:
                                        p_best_box_info = [cls_num, x1, y1, x2, y2, id, conf]
                                    else:
                                        pass

                                    same_prev_p_iou = p_iou
                                    p_prev_box_info = p_best_box_info.copy()
                                    p_prev_id = p_best_box_info[5]
                            
                            elif f_iou == p_iou:
                                pass
                            elif f_iou < p_iou:
                                if id == p_prev_id:
                                    p_best_box_info_tmp = newframe_or_not(new_frame, p_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                                    p_best_box_info = p_best_box_info_tmp.copy()

                                    p_prev_box_info = p_best_box_info.copy()
                                    p_prev_id = p_best_box_info[5]
                                else:
                                    # iou 검사해서 높으면 계속 f에 계속 갱신  
                                    prev_f_xyxy = f_prev_box_info[1:5].copy()
                                    now_f_xyxy = [x1, y1, x2, y2]
                                    f_iou = calculate_iou(prev_f_xyxy, now_f_xyxy)

                                    if f_iou >= same_prev_f_iou:
                                        f_best_box_info = [cls_num, x1, y1, x2, y2, id, conf]
                                    else:
                                        pass

                                    same_prev_f_iou = f_iou
                                    f_prev_box_info = f_best_box_info.copy()
                                    f_prev_id = f_best_box_info[5]          

                                
                        else:
                            if id == f_prev_id:    
                                f_best_box_info_tmp = newframe_or_not(new_frame, f_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                                f_best_box_info = f_best_box_info_tmp.copy()

                                f_prev_box_info = f_best_box_info.copy()
                                f_prev_id = f_best_box_info[5]

                            if  id == p_prev_id:
                                p_best_box_info_tmp = newframe_or_not(new_frame, p_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                                p_best_box_info = p_best_box_info_tmp.copy()

                                p_prev_box_info = p_best_box_info.copy()
                                p_prev_id = p_best_box_info[5]

                    elif f_prev_id in ids:
                        if id == f_prev_id:
                            f_best_box_info_tmp = newframe_or_not(new_frame, f_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                            f_best_box_info = f_best_box_info_tmp.copy()
                            
                            f_prev_box_info = f_best_box_info.copy()
                            f_prev_id = f_best_box_info[5]
                        else:
                            # iou 검사해서 높으면 계속 p에 계속 갱신
                            prev_p_xyxy = p_prev_box_info[1:5].copy()
                            now_p_xyxy = [x1, y1, x2, y2]
                            p_iou = calculate_iou(prev_p_xyxy, now_p_xyxy)

                            if p_iou >= prev_p_iou:
                                p_best_box_info = [cls_num, x1, y1, x2, y2, id, conf]
                            else:
                                pass

                            prev_p_iou = p_iou
                            p_prev_box_info = p_best_box_info.copy()
                            p_prev_id = p_best_box_info[5]
                                                                                
                    elif p_prev_id in ids:
                        if  id == p_prev_id:
                            p_best_box_info_tmp = newframe_or_not(new_frame, p_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                            p_best_box_info = p_best_box_info_tmp.copy()

                            p_prev_box_info = p_best_box_info.copy()
                            p_prev_id = p_best_box_info[5]

                        else:
                            # iou 검사해서 높으면 계속 f에 계속 갱신  
                            prev_f_xyxy = f_prev_box_info[1:5].copy()
                            now_f_xyxy = [x1, y1, x2, y2]
                            f_iou = calculate_iou(prev_f_xyxy, now_f_xyxy)

                            if f_iou >= prev_f_iou:
                                f_best_box_info = [cls_num, x1, y1, x2, y2, id, conf]
                            else:
                                pass

                            prev_f_iou = f_iou
                            f_prev_box_info = f_best_box_info.copy()
                            f_prev_id = f_best_box_info[5]                                           
                    else:
                        prev_f_xyxy = f_prev_box_info[1:5].copy()
                        prev_p_xyxy = p_prev_box_info[1:5].copy()
                        now_xyxy = [x1, y1, x2, y2]

                        f_iou = calculate_iou(prev_f_xyxy, now_xyxy)
                        p_iou = calculate_iou(prev_p_xyxy, now_xyxy)

                        if f_iou > p_iou:
                            f_best_box_info_tmp = newframe_or_not(new_frame, f_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                            f_best_box_info = f_best_box_info_tmp.copy()

                            prev_f_iou = f_iou
                            f_prev_box_info = f_best_box_info.copy()
                            f_prev_id = f_best_box_info[5]
                        
                        elif f_iou == p_iou:
                            pass
                        elif f_iou < p_iou:
                            p_best_box_info_tmp = newframe_or_not(new_frame, p_best_box_info, cls_num, x1, y1, x2, y2, id, conf)
                            p_best_box_info = p_best_box_info_tmp.copy()

                            prev_p_iou = p_iou
                            p_prev_box_info = p_best_box_info.copy()
                            p_prev_id = p_best_box_info[5]

                # best값을 csv에 작성
                data = f_best_box_info + p_best_box_info
                hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)

                prev_f_iou = 0.0
                prev_p_iou = 0.0

                f_prev_id = f_best_box_info[5]
                p_prev_id = p_best_box_info[5]
                
                frame_num = frame_num + 1
                new_frame = True

                # best 값 초기화
                f_best_box_info = ["f", 0, 0, 0, 0, f_prev_id, 0]
                p_best_box_info = ["p", 0, 0, 0, 0, p_prev_id, 0]

            else: # ID가 없으면 문구 출력하기
                print("Tracking IDs not available.")

                f_best_box_info = ["f", 0, 0, 0, 0, f_prev_id, 0]
                p_best_box_info = ["p", 0, 0, 0, 0, p_prev_id, 0]
                data = f_best_box_info + p_best_box_info
                hyotrack_csv_save_table(tmp_save_folder, csv_file_name, header, frame_num, data)    

                new_frame = True                                    
            
                frame_num = frame_num + 1

    # 보간법 수행
    tmp_file_path = os.path.join(tmp_save_folder, csv_file_name) # CSV 파일 경로

    writed_header, interpolation_data = read_csv_with_header(tmp_file_path)
    f_results = find_rows_with_value(interpolation_data, 'f', 1) # tmp에 저장한 data, 찾고자 하는 값, 검사할 열의 인덱스 (0 기반) 순서
    p_results = find_rows_with_value(interpolation_data, 'p', 8)

    # 보간 처리
    f_complete_data = process_results(f_results, interpolation_data, 2, 5, 0)
    f_p_complete_data = process_results(p_results, f_complete_data, 9, 12, 0)

    # CSV 파일에 헤더와 데이터를 덮어쓰기
    write_csv_with_header(save_folder, csv_file_name, writed_header, f_p_complete_data)
    
    # results 초기화
    results.clear()

if __name__ == "__main__":

    # tracking에 사용할 training 모델의 weight 파일 경로를 지정
    model_path = r"D:\COS_YOLOv8\240529_COS_bbox\240606_best.pt"
    # model = YOLO(r"D:\COS_YOLOv8\240529_COS_bbox\240602_best.pt")
    # model = YOLO(r"E:\1. experiment\COS\4. YOLOv8\240810_best.pt")
    # model = YOLO(r"E:\1. experiment\COS\4. YOLOv8\240810_last.pt")

    # Video file을 불러오기 위한 경로 지정
    # experiment_path = r"D:\tmp\#analysis\trim" ## 절대경로
    experiment_path = r"D:\tmp\#analysis(2)\trim" ## 절대경로
    group_folderlist = os.listdir(experiment_path) # experiment_path에서 그룹 폴더 목록을 가져옴

    # 결과를 저장할 경로
    save_folder = make_folder(experiment_path, "#data") # csv 저장 경로
    delete_folder_contents(save_folder) # save_foler 비우기 (반복실행시 앞선 결과 csv에 덮어쓰지 않도록)

    tmp_save_folder = make_folder(save_folder, "tmp_csv") # tmp 파일을 저장할 경로    
    ## botsort_video_save_folder = make_folder(save_folder, "#botsort_video") # borsort 실행 video 저장 경로
    ## hyotrack_video_save_forder = make_folder(save_folder, "#hyosort_video") # hyotrack 실행 video 저장 경로
    fail_list = [] # 새로운 fail_list를 생성

    # 각 그룹 폴더에 대한 반복 처리
    for group_foldername in group_folderlist: # Control, Toy, Pair 
        group_folderpath = os.path.join(experiment_path, group_foldername) # 개별 그룹 폴더의 전체 경로 생성

        if group_foldername == "#data":
            pass
        elif group_foldername == "trim(R)":
            file_list = os.listdir(group_folderpath) # 서브 폴더 내의 모든 파일 목록을 가져옴

            # 각 파일(동영상)에 대한 반복 처리
            for video_name_with_ext in file_list: 
                video_name = os.path.splitext(video_name_with_ext)[0] # 파일 이름에서 확장자를 제거하여 기본 파일 이름을 추출
                                    
                if os.path.splitext(video_name_with_ext)[1].lower() == '.mp4':  # 동영상 파일만 처리
                    # 분석할 동영상의 경로
                    video_path = os.path.join(group_folderpath, video_name_with_ext)
                    
                    # 동영상을 저장할 경로
                    ## borsort_video_output_path = os.path.join(botsort_video_save_folder, video_name + '_botsort.mp4') # borsort 저장할 video 이름
                    ## hyotrack_video_output_path = os.path.join(hyotrack_video_save_forder, video_name + '_hyosort.mp4') # hyotrack 저장할 video 이름
                    
                    # 저장할 csv 파일의 이름
                    csv_file_name = video_name + '.csv'

                    # video를 불러오기 위한 정보 지정
                    cap = cv2.VideoCapture(video_path)

                    # new_frame key 초기화
                    new_frame = None

                    if cap.isOpened(): # 동영상이 열리면 분석 수행
                        try:
                            # 비디오 속성 확인
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            # 결과 동영상 속성설정
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            ## botsort_out = cv2.VideoWriter(borsort_video_output_path, fourcc, fps, (width, height)) # botsort out 정보
                            ## hyotrack_out = cv2.VideoWriter(hyotrack_video_output_path, fourcc, fps, (width, height)) # hyotrack out 정보
                            
                            # 이전 프레임의 정보를 저장할 변수 초기화
                            previous_frame = None

                            # frame number를 세어줄 변수 초기화
                            frame_num = 0
                            new_frame = None

                            # object detection에 실패한 frame number를 저장할 새로운 리스트 생성
                            fail_frame_num_list = []

                            # header 지정
                            else_header = ['frame_number', 'cls_num', 'x1', 'y1', 'x2', 'y2', 'id', 'conf']

                            # control rat(one rat)에 대한 분석 수행 - control은 모든 session에서 rat만 한마리이기때문에 따로 비교하지 않음
                            one_rat(new_frame, tmp_save_folder, csv_file_name, else_header, frame_num, save_folder)

                            # cap을 닫아주는 과정 (종료를 위해)
                            cap.release()
                            cv2.destroyAllWindows()
                        except Exception as e: # error 발생시
                            error_message = str(e)
                            print(f"Error processing video: {video_path}, Error: {error_message}")
                            fail_list.append((video_path, error_message))
                        finally:
                            cap.release()
                            cv2.destroyAllWindows()
                    else: # 동영상이 안열리면
                        error_message = "Cannot open video file"
                        print(f"Error opening video: {video_path}, Error: {error_message}")
                        fail_list.append((video_path, error_message))                
        else:
            session_folderlist = os.listdir(group_folderpath) # 해당 그룹 폴더 내의 모든 하위 폴더(지역) 목록을 가져옴

            # 각 session 폴더에 대한 반복 처리
            for session_foldername in session_folderlist: # H, L1, L2, Lc, R
                session_path = os.path.join(group_folderpath, session_foldername) # 서브 폴더의 전체 경로 생성
                file_list = os.listdir(session_path) # 서브 폴더 내의 모든 파일 목록을 가져옴

                # 각 파일(동영상)에 대한 반복 처리
                for video_name_with_ext in file_list: 
                    video_name = os.path.splitext(video_name_with_ext)[0] # 파일 이름에서 확장자를 제거하여 기본 파일 이름을 추출
                                        
                    if os.path.splitext(video_name_with_ext)[1].lower() == '.mp4':  # 동영상 파일만 처리
                        # 분석할 동영상의 경로
                        video_path = os.path.join(session_path, video_name_with_ext)
                        
                        # 동영상을 저장할 경로
                        ## borsort_video_output_path = os.path.join(botsort_video_save_folder, video_name + '_botsort.mp4') # borsort 저장할 video 이름
                        ## hyotrack_video_output_path = os.path.join(hyotrack_video_save_forder, video_name + '_hyosort.mp4') # hyotrack 저장할 video 이름
                        
                        # 저장할 csv 파일의 이름
                        csv_file_name = video_name + '.csv'

                        # video를 불러오기 위한 정보 지정
                        cap = cv2.VideoCapture(video_path)

                        # new_frame key 초기화
                        new_frame = None

                        if cap.isOpened(): # 동영상이 열리면 분석 수행
                            try:
                                # 비디오 속성 확인
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                                # 결과 동영상 속성설정
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                ## botsort_out = cv2.VideoWriter(borsort_video_output_path, fourcc, fps, (width, height)) # botsort out 정보
                                ## hyotrack_out = cv2.VideoWriter(hyotrack_video_output_path, fourcc, fps, (width, height)) # hyotrack out 정보
                                
                                # 이전 프레임의 정보를 저장할 변수 초기화
                                previous_frame = None

                                # frame number를 세어줄 변수 초기화
                                frame_num = 0
                                new_frame = None

                                # object detection에 실패한 frame number를 저장할 새로운 리스트 생성
                                fail_frame_num_list = []

                                if group_foldername == "Control":
                                    # header 지정
                                    C_header = ['frame_number', 'cls_num', 'x1', 'y1', 'x2', 'y2', 'id', 'conf']

                                    # control rat(one rat)에 대한 분석 수행 - control은 모든 session에서 rat만 한마리이기때문에 따로 비교하지 않음
                                    one_rat(new_frame, tmp_save_folder, csv_file_name, C_header, frame_num, save_folder)

                                elif group_foldername == "Toy":
                                    # header 지정
                                    T_header = ['frame_number', 'rat_cls_num', 'rat_x1', 'rat_y1', 'rat_x2', 'rat_y2', 'rat_id', 'rat_conf', 'toy_cls_num', 'toy_x1', 'toy_y1', 'toy_x2', 'toy_y2', 'toy_id', 'toy_conf']

                                    if session_foldername == "H" or session_foldername == "Lc" or session_foldername == "R":
                                        one_rat(new_frame, tmp_save_folder, csv_file_name, T_header, frame_num, save_folder)
                                    elif session_foldername == "L1" or session_foldername == "L2":
                                        one_rat_one_toy(new_frame, tmp_save_folder, csv_file_name, T_header, frame_num, save_folder)
                                    else:
                                        session_foldername_error_message = "session_foldername_error"
                                        print(f"Error processing video: {video_path}, Error: {session_foldername_error_message}")
                                        fail_list.append((video_path, session_foldername_error_message))                                    

                                elif group_foldername == "Pair":
                                    # header 지정
                                    P_header = ['frame_number', 'F_cls_num', 'F_x1', 'F_y1', 'F_x2', 'F_y2', 'F_id', 'F_conf', 'P_cls_num', 'P_x1', 'P_y1', 'P_x2', 'P_y2', 'P_id', 'P_conf']

                                    if session_foldername == "H" or session_foldername == "Lc" or session_foldername == "R":
                                        one_rat(new_frame, tmp_save_folder, csv_file_name, P_header, frame_num, save_folder)
                                    elif session_foldername == "L1" or session_foldername == "L2":
                                        two_rats(new_frame, width, tmp_save_folder, csv_file_name, P_header, frame_num, save_folder)
                                    else:
                                        session_foldername_error_message = "session_foldername_error"
                                        print(f"Error processing video: {video_path}, Error: {session_foldername_error_message}")
                                        fail_list.append((video_path, session_foldername_error_message))        

                                elif group_foldername == "Friend":
                                    # header 지정
                                    F_header = ['frame_number', 'cls_num', 'x1', 'y1', 'x2', 'y2', 'id', 'conf']

                                    # friend rat(one rat)에 대한 분석 수행 - friend는 H 세션만 따로 문석하고 L1, L2는 pair과 함께 분석함(H는 rat만 한마리)
                                    one_rat(new_frame, tmp_save_folder, csv_file_name, F_header, frame_num, save_folder)
                                
                                else:
                                    print("group folder name error")

                                # cap을 닫아주는 과정 (종료를 위해)
                                cap.release()
                                cv2.destroyAllWindows()
                            except Exception as e: # error 발생시
                                error_message = str(e)
                                print(f"Error processing video: {video_path}, Error: {error_message}")
                                fail_list.append((video_path, error_message))
                            finally:
                                cap.release()
                                cv2.destroyAllWindows()
                        else: # 동영상이 안열리면
                            error_message = "Cannot open video file"
                            print(f"Error opening video: {video_path}, Error: {error_message}")
                            fail_list.append((video_path, error_message))                

    # 실패 목록을 txt 파일로 저장
    save_fail_list(fail_list, save_folder)

    print("done")