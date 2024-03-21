# image_processor.py
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImageProcessor:
    def __init__(self,relative_path):
        self.relative_path = relative_path
        self.Blue = (255,0,0)
        self.Green = (0,255,0)
        self.Red = (0,0,255) 
        self.Black = (0,0,0)
        self.White = (255,255,255)
        self.DRAW_BG = {'color':self.Black,'val':0}
        self.DRAW_FG = {'color':self.White,'val':1}
        self.rect = (0,0,1,1)
        self.drawing = False
        self.rectangle = False
        self.rect_over = False
        self.rect_or_mask = 100
        self.value = self.DRAW_FG
        self.thickness = 11
        
        # 이전 상태를 저장하기 위한 변수
        self.prev_img = None
        self.prev_mask = None
        self.saved_mask = None
        self.next = None
        
        self.img = cv2.imread(relative_path)
        self.img_temp = self.img.copy()
        
        self.mask = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.output = np.zeros(self.img.shape,np.uint8)
        
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)
        
        self.ix = None
        self.iy = None
        self.org_path = './output/org'
        self.seg_path = './output/seg'
        
    def run(self):
        cv2.namedWindow('input', cv2.WINDOW_NORMAL)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('input',self.onMouse,param=(self.img,self.img_temp))
        cv2.moveWindow('input',self.img.shape[1]+10,90)
        print('오른쪽 마우스 버튼을 누르고 영역을 지정한 후 n을 누르시오')
        
        while True:
            cv2.imshow('output',self.output)
            cv2.imshow('input',self.img)

            k = cv2.waitKey(1) & 0xFF
            
            # esc키누르면 종료
            if k == 27:  
                exit()
                break
            
            # ']' 키를 누르면 thickness 증가
            if k == ord(']'):
                self.thickness += 3
                print(f'현재 두께: {self.thickness}')

            # '[' 키를 누르면 thickness 감소, 단, thickness는 1보다 작아질 수 없음
            if k == ord('[') and self.thickness > 1:
                self.thickness -= 3
                print(f'현재 두께: {self.thickness}')

            if k == ord('0'):
                print('왼쪽 마우스로 제거할 부분을 표시한 후 n을 누르세요')
                self.value = self.DRAW_BG
            if k == ord('1'):
                print('왼쪽 마우스로 복원할 부분을 표시한 후 n을 누르세요')
                self.value = self.DRAW_FG
        
            elif k == ord('n'):
                # 이전상태 저장
                self.prev_img = self.img_temp.copy()
                self.prev_mask = self.mask.copy()
                
                self.apply_grabcut()

                print('0:제거할 배경선택, 1:복원할 전경선택, n:적용하기')
                print('s:저장 후 계속, r:리셋, b:되돌리기, esc:완전종료')
                print('크기조절 키우기 ], 줄이기 [')
                
                # 업데이트된 마스크를 흑백으로 변환하여 mask_temp에 저장
                self.mask_temp = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8') * 255
                
                
                # 모폴리지 연산
                self.apply_morphology()
                
                # 경계선 평활화
                self.apply_approxPoly()
                
                # mask_temp를 'mask' 윈도우에 표시
                cv2.imshow('mask', self.mask_temp)
                
                # 저장하기누르면 현재 창에보이는 결과물이 저장될것임
                self.saved_mask = self.mask_temp.copy()
                # 전경을 마스크를 기반으로 다시 추출해준다
                self.output = cv2.bitwise_and(self.img_temp, self.img_temp, mask=self.mask_temp)
            

            elif k == ord('s'):  # 's' 키를 누르면 원본이미지와 마스크가 저장됨
                self.apply_save_mask()
                break
                
            # 전경을 흑백으로 추출
            self.mask_temp = np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8') * 255
            self.output = cv2.bitwise_and(self.img_temp,self.img_temp,mask=self.mask_temp)

        cv2.destroyAllWindows()
        if self.next:
            from interface import load_and_process_image
            relative_path = load_and_process_image()
            ImageProcessor(relative_path).run()
        
    # 배경모델, 전경모델 생성
    def apply_grabcut(self):
        if self.rect_or_mask == 0:
            cv2.grabCut(self.img_temp,self.mask,self.rect,self.bgdModel,self.fgdModel,1,cv2.GC_INIT_WITH_RECT)
            self.rect_or_mask = 1
        elif self.rect_or_mask == 1:
            cv2.grabCut(self.img_temp,self.mask,self.rect,self.bgdModel,self.fgdModel,1,cv2.GC_INIT_WITH_MASK)
        return

    def apply_morphology(self):
        # 커널 크기
        kernel = np.ones((4,4), np.uint8)
        # Opening 연산으로 노이즈 제거
        opening = cv2.morphologyEx(self.mask_temp, cv2.MORPH_OPEN, kernel)
        # Closing 연산으로 작은 구멍 메우기 
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        self.mask_temp = closing
        return

    def apply_approxPoly(self):
        # 경계를 찾는다
        contours, hierarchy = cv2.findContours(self.mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 각 컨투어에 대해 근사화를 수행하고 마스크에 그린다
        # mask_temp로 저장한다 
        self.mask_temp = np.zeros_like(self.mask_temp)
        for cnt in contours:   
            #approxPolyDP의 두번째 매개변수를 적절히 조절하면 경계가 단순화된다
            approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
            # 흑백만 걸러서 노이즈 방지
            cv2.drawContours(self.mask_temp, [approx], 0, (255), thickness=cv2.FILLED)
        return 

    def apply_save_mask(self): 
        filename, file_extension = os.path.splitext(os.path.basename(self.relative_path))
        output_path = os.path.join(self.seg_path, filename + file_extension)

        # 저장경로 잇는지 확인
        if not os.path.exists(self.seg_path):
            os.makedirs(self.seg_path)

        # 사이즈 바꾸기
        self.saved_mask = cv2.resize(self.saved_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        # 마스크 저장
        cv2.imwrite(output_path, self.saved_mask)
        print(f'마스크 이미지를 {output_path}로 저장했습니다.')

        # org 변환해서 저장
        output_path = os.path.join(self.org_path, filename + file_extension)
        if not os.path.exists(self.org_path):
            os.makedirs(self.org_path)
            
        self.img_temp = cv2.resize(self.img_temp, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, self.img_temp)
        print(f'오리지날 이미지를 {output_path}로 저장했습니다.')
        self.next = 1
        return

    def onMouse(self,event,x,y,flags,params):
        
        # 연필 크기 보기
        if event == cv2.EVENT_MOUSEMOVE and not self.drawing:
            self.img = self.img_temp.copy()
            # 현재 마우스 위치에 따라 원을 그립니다. 이 원은 현재 thickness 값을 반영합니다.
            cv2.circle(self.img, (x, y), self.thickness, (0, 0, 255), 2)  # 마우스 위치에 원을 그림

        if event == cv2.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix,self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectangle:
                self.img = self.img_temp.copy()
                cv2.rectangle(self.img,(self.ix,self.iy),(x,y),self.Red,2)
                self.rect = (min(self.ix,x),min(self.iy,y),abs(self.ix-x),abs(self.iy-y))
                self.rect_or_mask = 0

        elif event == cv2.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True

            cv2.rectangle(self.img,(self.ix,self.iy),(x,y),self.Red,2)
            self.rect = (min(self.ix,x),min(self.iy,y),abs(self.ix-x),abs(self.iy-y))
            self.rect_or_mask = 0
            print('n:적용하기')

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.rect_over:
                print('마우스 왼쪽을 누른채로 전경이 되는 부분을 선택하시오')
            else:
                self.drawing = True
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)
        return