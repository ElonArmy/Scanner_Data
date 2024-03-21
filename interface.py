#UI
import tkinter as tk
from tkinter import filedialog
import os
from image_processor import ImageProcessor

# 파일선택창    
def select_image_file():
    root = tk.Tk()
    root.withdraw()  # Tkinter 창을 띄우지 않고 파일 선택 대화 상자만 사용
    file_path = filedialog.askopenfilename()  # 파일 선택 창을 띄움
    return file_path

# 이미지 선택해서 경로 가지고와서 그랩컷 실행
def load_and_process_image():
    while True:
        image_path = select_image_file()
        if image_path:  # 사용자가 파일을 선택한 경우
            # 현재 작업 디렉토리의 절대 경로
            current_dir = os.path.abspath(os.getcwd())  
            # 상대 경로 계산
            relative_path = os.path.relpath(image_path, current_dir)  
            print(f"선택된 파일: {image_path}")
            print(f"전달 경로: {relative_path}")
            # 그랩컷 프로세서에 전달할 경로 반환
            return relative_path
        else:
            print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
            break