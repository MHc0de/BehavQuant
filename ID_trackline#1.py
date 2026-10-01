import cv2
import csv
import os
import shutil
from tqdm import tqdm

def read_csv_with_header(file_path):
    # CSV 파일을 읽어 첫 줄은 헤더로, 나머지 줄은 데이터로 반환하는 함수
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        data_list = [row for row in csvreader]
    return header, data_list

def make_folder(data_path, save_folder_name):
    # 경로 상에 폴더가 존재하는지 확인 후 없다면 폴더 생성
    save_path = os.path.join(data_path, save_folder_name)
    if not os.path.exists(save_path): # 경로가 존재하지 않으면
        os.makedirs(save_path, exist_ok=True) # 경로 생성 (mkdir은 마지막 경로만 만들음, makedirs는 경로가 없으면 상위까지 전부 만들어줌)
    else:
        print(f"경로 '{save_path}'이(가) 이미 존재합니다.")
    return save_path

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

if __name__ == "__main__":

    # Video file을 불러오기 위한 경로 지정
    experiment_path = r"C:\Users\ddogd\Desktop\250315\COS_final_final_analysis_pair_line(representative)" ## 절대경로
    group_folderlist = os.listdir(experiment_path) # experiment_path에서 그룹 폴더 목록을 가져옴

    # csv결과가 저장된 경로
    save_folder = os.path.join(experiment_path, "#data") # csv 저장 경로

    hyotrackline_image_save_forder = make_folder(save_folder, "#hyosort_line_image") # hyotrack 실행 video 저장 경로

    fail_list = [] # 새로운 fail_list를 생성

    # 각 그룹 폴더에 대한 반복 처리
    for group_foldername in group_folderlist: # Control, Toy, Pair 
        group_folderpath = os.path.join(experiment_path, group_foldername) # 개별 그룹 폴더의 전체 경로 생성
        session_folderlist = os.listdir(group_folderpath) # 해당 그룹 폴더 내의 모든 하위 폴더(지역) 목록을 가져옴
        
        if group_foldername == "#data":
            pass
        else:
            # 각 session 폴더에 대한 반복 처리
            for session_foldername in session_folderlist: # H, L1, L2, Lc, R
                session_path = os.path.join(group_folderpath, session_foldername) # 서브 폴더의 전체 경로 생성
                file_list = os.listdir(session_path) # 서브 폴더 내의 모든 파일 목록을 가져옴

                # 각 파일(동영상)에 대한 반복 처리
                for video_name_with_ext in tqdm(file_list, desc = group_foldername + " " + session_foldername, position = 0): 
                    video_name = os.path.splitext(video_name_with_ext)[0] # 파일 이름에서 확장자를 제거하여 기본 파일 이름을 추출             
                                            
                    if os.path.splitext(video_name_with_ext)[1].lower() == '.mp4':  # 동영상 파일만 처리
                        # tracking을 수행할 동영상의 경로
                        video_path = os.path.join(session_path, video_name_with_ext)

                        # 정보를 가져올 csv파일의 경로
                        csv_path = os.path.join(save_folder, video_name) + '.csv'
                        
                        # # 개별 동영상을 저장할 경로
                        # hyotrackline_image_output_path = os.path.join(hyotrackline_image_save_forder, video_name + '_hyosort.mp4') # hyotrack 저장할 video 이름
                        
                        # video를 불러오기 위한 정보 지정
                        cap = cv2.VideoCapture(video_path)

                        # 이전 프레임의 중심 좌표를 저장할 변수 초기화
                        prev_f_center = None
                        prev_p_center = None

                        if cap.isOpened(): # 동영상이 열리면 분석 수행
                            try:
                                # 첫 번째 프레임 저장
                                ret, first_frame = cap.read()

                                if ret:
                                    first_frame_path = os.path.join(hyotrackline_image_save_forder, video_name + '_first_frame.jpg')
                                    cv2.imwrite(first_frame_path, first_frame)
                                    first_frame_F = first_frame.copy()
                                    first_frame_P = first_frame.copy()
                                else:
                                    raise Exception("첫 번째 프레임을 읽어오지 못했습니다.")            
                                
                                fail_frame_num_list = []
                                
                                if group_foldername == "Control":
                                    pass
                                elif group_foldername == "Toy":

                                    # data 폴더에서 파일이름이랑 같은 csv파일 불러오기
                                    header, data = read_csv_with_header(csv_path)

                                    n = len(data)
                                    frame_num = 0

                                    # 비디오 프레임에 대한 수행 과정
                                    # 무한루프

                                    with tqdm(total = n-1, desc = video_name_with_ext, position = 1, leave = False) as pbar:
                                        
                                        while frame_num < n:

                                            # tqdm 객체 선언
                                            pbar.update(1)

                                            f_box_info = data[frame_num][1:7].copy()  # 데이터 복사
                                            p_box_info = data[frame_num][8:14].copy()  # 데이터 복사
                                            
                                            f_xyxy = [int(float(val)) for val in f_box_info[1:5]]
                                            p_xyxy = [int(float(val)) for val in p_box_info[1:5]]

                                            f_center = (int((f_xyxy[0] + f_xyxy[2]) / 2), int((f_xyxy[1] + f_xyxy[3]) / 2))
                                            p_center = (int((p_xyxy[0] + p_xyxy[2]) / 2), int((p_xyxy[1] + p_xyxy[3]) / 2))
                                            
                                            f_box_color = (71, 99, 255)
                                            p_box_color = (235, 206, 135)
                                            
                                            f_group_name = "Friend"
                                            p_group_name = "Pair"

                                            f_label = f"{f_group_name}"
                                            p_label = f"{p_group_name}"

                                            # 추적 경로 그리기
                                            if frame_num == 0:
                                                pass
                                                cv2.line(first_frame_F, f_center, f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, p_center, p_center, p_box_color, 1, cv2.LINE_AA)

                                            elif frame_num == n-1:
                                                cv2.line(first_frame_F, prev_f_center, prev_f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, prev_f_center, prev_p_center, p_box_color, 1, cv2.LINE_AA)
                                            else:
                                                cv2.line(first_frame_F, f_center, prev_f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, p_center, prev_p_center, p_box_color, 1, cv2.LINE_AA)
                                                

                                            # 이전 프레임의 중심 좌표를 저장할 변수 초기화
                                            prev_f_center = f_center
                                            prev_p_center = p_center

                                            frame_num += 1
                                    
                                        pbar.close()

                                        # 종료를 위한 과정
                                        cap.release()
                                        cv2.destroyAllWindows()
                                    first_frame_path_F = os.path.join(hyotrackline_image_save_forder, video_name + '_first_frame_F.jpg')
                                    first_frame_path_P = os.path.join(hyotrackline_image_save_forder, video_name + '_first_frame_P.jpg')

                                    cv2.imwrite(first_frame_path_F, first_frame_F)
                                    cv2.imwrite(first_frame_path_P, first_frame_P)
                                    
                                    pass
                                elif group_foldername == "Pair":

                                    # data 폴더에서 파일이름이랑 같은 csv파일 불러오기
                                    header, data = read_csv_with_header(csv_path)

                                    n = len(data)
                                    frame_num = 0

                                    # 비디오 프레임에 대한 수행 과정
                                    # 무한루프

                                    with tqdm(total = n-1, desc = video_name_with_ext, position = 1, leave = False) as pbar:
                                        
                                        while frame_num < n:

                                            # tqdm 객체 선언
                                            pbar.update(1)

                                            f_box_info = data[frame_num][1:7].copy()  # 데이터 복사
                                            p_box_info = data[frame_num][8:14].copy()  # 데이터 복사
                                            
                                            f_xyxy = [int(float(val)) for val in f_box_info[1:5]]
                                            p_xyxy = [int(float(val)) for val in p_box_info[1:5]]

                                            f_center = (int((f_xyxy[0] + f_xyxy[2]) / 2), int((f_xyxy[1] + f_xyxy[3]) / 2))
                                            p_center = (int((p_xyxy[0] + p_xyxy[2]) / 2), int((p_xyxy[1] + p_xyxy[3]) / 2))
                                            
                                            f_box_color = (71, 99, 255)
                                            p_box_color = (235, 206, 135)
                                            
                                            f_group_name = "Friend"
                                            p_group_name = "Pair"

                                            f_label = f"{f_group_name}"
                                            p_label = f"{p_group_name}"

                                            # 추적 경로 그리기
                                            if frame_num == 0:
                                                pass
                                                cv2.line(first_frame_F, f_center, f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, p_center, p_center, p_box_color, 1, cv2.LINE_AA)

                                            elif frame_num == n-1:
                                                cv2.line(first_frame_F, prev_f_center, prev_f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, prev_f_center, prev_p_center, p_box_color, 1, cv2.LINE_AA)
                                            else:
                                                cv2.line(first_frame_F, f_center, prev_f_center, f_box_color, 1, cv2.LINE_AA)
                                                cv2.line(first_frame_P, p_center, prev_p_center, p_box_color, 1, cv2.LINE_AA)
                                                

                                            # 이전 프레임의 중심 좌표를 저장할 변수 초기화
                                            prev_f_center = f_center
                                            prev_p_center = p_center

                                            frame_num += 1
                                    
                                        pbar.close()

                                        # 종료를 위한 과정
                                        cap.release()
                                        cv2.destroyAllWindows()
                                    first_frame_path_F = os.path.join(hyotrackline_image_save_forder, video_name + '_first_frame_F.jpg')
                                    first_frame_path_P = os.path.join(hyotrackline_image_save_forder, video_name + '_first_frame_P.jpg')

                                    cv2.imwrite(first_frame_path_F, first_frame_F)
                                    cv2.imwrite(first_frame_path_P, first_frame_P)

                                else:  
                                    error_message = "group folder name error"
                                    print(f"Error processing video: {video_path}, Error: {error_message}")
                                    fail_list.append((video_path, error_message))
                                
                            except Exception as e: # error 발생시
                                error_message = str(e)
                                print(f"Error processing video: {video_path}, Error: {error_message}")
                                fail_list.append((video_path, error_message))

    print("done")