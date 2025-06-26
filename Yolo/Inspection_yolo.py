import torch
import os
import shutil
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = "BASE_DIR"
MODEL_PATH = os.path.join(BASE_DIR, "MODEL_PATH")
best_model_path = os.path.join(MODEL_PATH, 'weights', 'best.pt')
input_dir = os.path.join(BASE_DIR, "input_data_dir")
detected_dir = os.path.join(BASE_DIR, "detection")
undetected_dir = os.path.join(BASE_DIR, "non-detection")

def main():
    # 결과 디렉토리 생성
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(undetected_dir, exist_ok=True)
    
    # 모델 로드
    if not os.path.exists(best_model_path):
        logger.error(f"모델 파일이 존재하지 않습니다: {best_model_path}")
        return
        
    logger.info(f"모델 로드 중: {best_model_path}")
    model = YOLO(best_model_path)
    
    # 이미지 파일 찾기 (단일 패턴으로 효율적으로 처리)
    image_paths = list(Path(input_dir).glob("**/*.jpg")) + list(Path(input_dir).glob("**/*.png"))
    logger.info(f"총 {len(image_paths)}개의 이미지 파일을 찾았습니다.")
    
    detected_count = 0
    undetected_count = 0
    
    for img_path in image_paths:
        try:
            # 예측 수행
            results = model.predict(source=str(img_path), save=False, verbose=False, save_txt=False)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # 탐지된 이미지 저장
                save_path = os.path.join(detected_dir, img_path.name)
                shutil.copy(img_path, save_path)
                
                # 이미지 크기 가져오기 (정규화를 위해)
                with Image.open(img_path) as img:
                    w, h = img.size
                
                # 라벨 저장
                label_path = os.path.splitext(save_path)[0] + ".txt"
                with open(label_path, "w") as f:
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i])
                        x1, y1, x2, y2 = boxes.xyxy[i]
                        
                        # 중심좌표, width, height 계산 및 정규화
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                #logger.info(f"[✔] Detected & Saved: {img_path.name} + label")
                detected_count += 1
            else:
                shutil.copy(img_path, os.path.join(undetected_dir, img_path.name))
                #logger.info(f"[ ] Not Detected: {img_path.name}")
                undetected_count += 1
                
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {img_path.name} - {e}")
    
    logger.info(f"\n처리 완료! 탐지된 이미지: {detected_count}개, 미탐지 이미지: {undetected_count}개")

if __name__ == "__main__":
    main()
