import csv
import pandas as pd
import os
import math
from tqdm import tqdm

def make_folder(data_path, save_folder_name):
    # 경로 상에 폴더가 존재하는지 확인 후 없다면 폴더 생성
    save_path = os.path.join(data_path, save_folder_name)
    if not os.path.exists(save_path): # 경로가 존재하지 않으면
        os.makedirs(save_path, exist_ok=True) # 경로 생성 (mkdir은 마지막 경로만 만들음, makedirs는 경로가 없으면 상위까지 전부 만들어줌)
        print(f"경로 '{save_path}'을(를) 생성하였습니다.")
    else:
        print(f"경로 '{save_path}'이(가) 이미 존재합니다.")
    return save_path

def read_csv_with_header(file_path):
    # CSV 파일을 읽어 첫 줄은 헤더로, 나머지 줄은 데이터로 반환하는 함수
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data_list = [row for row in csvreader]
    return header, data_list

def in_same_room(box1, box2, center_x):
    # box는 (x1, y1, x2, y2)의 형태로 주어집니다.
    box1_x1 = int(box1[0])
    box1_y1 = int(box1[1])
    box1_x2 = int(box1[2])
    box1_y2 = int(box1[3])

    box2_x1 = int(box2[0])
    box2_y1 = int(box2[1])
    box2_x2 = int(box2[2])
    box2_y2 = int(box2[3])
    
    # 각각의 가운데 점을 구합니다.
    center1 = ((box1_x1 + box1_x2) / 2, (box1_y1 + box1_y2) / 2)
    center2 = ((box2_x1 + box2_x2) / 2, (box2_y1 + box2_y2) / 2)
    
    # 두 점이 주어진 center_x 기준으로 같은 쪽에 있는지 검사합니다.
    if (center1[0] < center_x and center2[0] < center_x):
        return 1, "W"
    elif (center1[0] > center_x and center2[0] > center_x):
        return 1, "B"
    else:
        return 0, 0

def distance_between_subject(box1, box2, pixel_514):

    box1_x1 = int(box1[0])
    box1_y1 = int(box1[1])
    box1_x2 = int(box1[2])
    box1_y2 = int(box1[3])

    box2_x1 = int(box2[0])
    box2_y1 = int(box2[1])
    box2_x2 = int(box2[2])
    box2_y2 = int(box2[3])
    
    # 각각의 가운데 점을 구합니다.
    f_center = ((box1_x1 + box1_x2) / 2, (box1_y1 + box1_y2) / 2)
    p_center = ((box2_x1 + box2_x2) / 2, (box2_y1 + box2_y2) / 2)

    distance_pixel = math.sqrt((f_center[0] - p_center[0]) ** 2 + (f_center[1] - p_center[1]) ** 2)
    distance_mm = distance_pixel * (514 / pixel_514)

    return distance_pixel, distance_mm

def csv_add_new_column(input_path, output_path, new_column_data, new_row_name):
    # 기존 CSV 파일 읽기

    df = pd.read_csv(input_path)

    # 새로운 데이터의 길이 가져오기
    new_length = len(new_column_data)

    # 데이터프레임의 인덱스를 새로운 길이에 맞게 확장
    df = df.reindex(range(new_length))

    # 새로운 열 추가
    df[f'{new_row_name}'] = new_column_data

    # 새로운 파일 또는 기존 파일에 작성하기
    df.to_csv(output_path, index=False)

def write_csv_with_header(output_folder, header, data):
    # CSV 파일을 저장하는 함수

    output_path = os.path.join(output_folder, "total_data") + ".csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(data)

if __name__ == "__main__":

    # file_dir_list = [r"E:\1. experiment\COS\4. YOLOv8\240901_analysis_pair\F\L1",
    #                 r"E:\1. experiment\COS\4. YOLOv8\240901_analysis_pair\F\L2",
    #                 r"E:\1. experiment\COS\4. YOLOv8\240901_analysis_pair\M\L1",
    #                 r"E:\1. experiment\COS\4. YOLOv8\240901_analysis_pair\M\L2"]
    file_dir_list = [r"D:\2. [Noh's lab]\2024 experiment\3. COS\1. experiment\4. YOLOv8\COS_final_final_analysis_toy_line\#data"]
    
    for file_dir in file_dir_list:

        csv_file_list = os.listdir(file_dir) # 서브 폴더 내의 모든 파일 목록을 가져옴
        sameroom_folder = make_folder(file_dir, "#same_room")

        total_list = []

        print(file_dir)

        # 각 파일에 대한 반복 처리
        for csv_file_name_with_ext in tqdm(csv_file_list): 
            csv_file_name = os.path.splitext(csv_file_name_with_ext)[0] # 파일 이름에서 확장자를 제거하여 기본 파일 이름을 추출
                                
            if os.path.splitext(csv_file_name_with_ext)[1].lower() == '.csv':  # csv 파일만 처리

                file_path = os.path.join(file_dir, csv_file_name) + '.csv'
                sameroom_path = os.path.join(sameroom_folder, csv_file_name) + '.csv'

                header, last_data = read_csv_with_header(file_path)

                sameroom_list = []
                room_color_list = []
                distance_pixel_list = []
                distance_mm_list = []

                sameroom_num = 0
                sameroom_W_num = 0
                sameroom_B_num = 0

                total_distance_pixel = 0
                total_distance_mm = 0

                near_40 = 0
                near_50 = 0
                near_60 = 0
                near_70 = 0
                near_80 = 0
                near_90 = 0

                near_0_to_20 = 0 
                near_20_to_40 = 0
                near_40_to_70 = 0
                near_70_to_125 = 0
                near_125_to_250 = 0
                near_250_to_500 = 0

                total_frame_num = len(last_data)

                for last_data_row in range(len(last_data)):
                    f_box_xyxy = last_data[last_data_row][2:6].copy()
                    p_box_xyxy = last_data[last_data_row][9:13].copy()
                    # center_x = center_x = width / 2
                    center_x = 1920 / 2
                    # pixel_514 = (1835 / 1920) * width 
                    pixel_514 = (1835 / 1920) * 1920

                    sameroom_value, room_color = in_same_room(f_box_xyxy, p_box_xyxy, center_x) # 1이면 같은방, 0이면 다른방

                    if sameroom_value == 1:
                        sameroom_num = sameroom_num + 1

                    if room_color == "W":
                        sameroom_W_num = sameroom_W_num + 1
                    elif room_color == "B":
                        sameroom_B_num = sameroom_B_num + 1
                    else:
                        pass

                    distance_pixel, distance_mm = distance_between_subject(f_box_xyxy, p_box_xyxy, pixel_514)

                    total_distance_pixel = total_distance_pixel + distance_pixel
                    total_distance_mm = total_distance_mm + distance_mm

                    if distance_mm <= 40:
                        near_40 = near_40 + 1                
                    if distance_mm <= 50:
                        near_50 = near_50 + 1
                    if distance_mm <= 60:
                        near_60 = near_60 + 1
                    if distance_mm <= 70:
                        near_70 = near_70 + 1
                    if distance_mm <= 80:
                        near_80 = near_80 + 1
                    if distance_mm <= 90:
                        near_90 = near_90 + 1

                    if distance_mm <= 20:
                        near_0_to_20 = near_0_to_20 + 1
                    elif distance_mm <= 40:
                        near_20_to_40 = near_20_to_40 + 1                 
                    elif distance_mm <= 70:
                        near_40_to_70 = near_40_to_70 + 1 
                    elif distance_mm <= 125:
                        near_70_to_125 = near_70_to_125 + 1 
                    elif distance_mm <= 250:
                        near_125_to_250 = near_125_to_250 + 1 
                    else: # 대략 500까지
                        near_250_to_500 = near_250_to_500 + 1 

                    sameroom_list.append(sameroom_value)
                    room_color_list.append(room_color)
                    distance_pixel_list.append(distance_pixel)
                    distance_mm_list.append(distance_mm)
                    
                sameroom_list.append(f'=sum(P2:P{(total_frame_num+1)})')
                sameroom_list.append('White_frame')
                sameroom_list.append('Black_frame')

                room_color_list.append('')
                room_color_list.append(f'=COUNTIF(Q2:Q{(total_frame_num+1)}, "W")')
                room_color_list.append(f'=COUNTIF(Q2:Q{(total_frame_num+1)}, "B")')

                distance_pixel_list.append(f'=average(R2:R{(total_frame_num+1)})')
                distance_pixel_list.append('')
                distance_pixel_list.append('')
                distance_mm_list.append(f'=average(S2:S{(total_frame_num+1)})')
                distance_mm_list.append('')
                distance_mm_list.append('')
                
                csv_add_new_column(file_path, sameroom_path, sameroom_list, 'in same room')
                csv_add_new_column(sameroom_path, sameroom_path, room_color_list, 'same room color')
                csv_add_new_column(sameroom_path, sameroom_path, distance_pixel_list, 'distance_pixel')
                csv_add_new_column(sameroom_path, sameroom_path, distance_mm_list, 'distance_mm')

                total_list_header = ['csv_file_name', 'sameroom_num', 'sameroom_num(%)','sameroom_W_num', 'sameroom_W_num(%)', 'sameroom_B_num', 'sameroom_B_num(%)', 
                                    'average_total_distance_pixel/total_frame_num', 'average_total_distance_mm',
                                    'near_40', 'near_50', 'near_60', 'near_70', 'near_80', 'near_90', 
                                    'near_0_to_20', 'near_20_to_40', 'near_40_to_70', 'near_70_to_125', 'near_125_to_250', 'near_250_to_500']

                total_list.append([csv_file_name, sameroom_num, (sameroom_num/total_frame_num)*100, sameroom_W_num, (sameroom_W_num/sameroom_num)*100, sameroom_B_num, (sameroom_B_num/sameroom_num)*100,
                                total_distance_pixel/total_frame_num, total_distance_mm/total_frame_num,
                                near_40, near_50, near_60, near_70, near_80, near_90, 
                                near_0_to_20, near_20_to_40, near_40_to_70, near_70_to_125, near_125_to_250, near_250_to_500])

        # print(total_list)
        write_csv_with_header(sameroom_folder, total_list_header, total_list)

