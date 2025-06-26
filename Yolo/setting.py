args = {
    # 일반 학습 설정
    'model': 'rtdetr-s.pt',          # 사용할 모델 (.pt or .yaml)
    'data': 'data.yaml',             # 학습용 데이터셋 구성 파일 (필수)
    'epochs': 100,                   # 학습 반복 횟수 (추천: 100~300)
    'batch': 1,                     # 배치 크기 (-1: 자동, 추천: GPU 메모리에 맞게 설정)
    'imgsz': 640,                    # 입력 이미지 크기 (정사각형 기준, 추천: 640~1280)
    'device': 0,                     # 학습에 사용할 GPU 번호 또는 'cpu'
    'workers': 8,                    # 데이터 로딩 워커 수 (추천: CPU 코어 수에 비례)
    'cache': False,                  # 이미지 캐싱 ('ram', 'disk', False)
    'resume': False,                 # 체크포인트 이어서 학습 여부
    'pretrained': True,              # 사전학습된 가중치 사용 여부
    'seed': 0,                       # 랜덤 시드 (재현성 확보)
    'deterministic': True,           # 완전한 재현성 확보 (속도는 다소 느림)
    'single_cls': False,            # 클래스 하나로 통합할지 여부 (binary task에 유용)
    'classes': [5],                 # 사용할 클래스 인덱스 목록 (None이면 전체)
    'fraction': 1.0,                 # 사용할 데이터셋 비율 (0~1)
    'amp': True,                     # 혼합 정밀도 (AMP) 사용 여부 (메모리 효율 및 속도 ↑)
    'multi_scale': False,            # 다중 해상도 학습 사용 여부 (imgsz ±50% 범위)
    'close_mosaic': 10,           # 마지막 N 에포크 동안 mosaic 중단 (ex: 10)
    'rect': False,                   # 직사각형 배치 사용 여부 (효율성 ↑)
    'time': 60,                      # 최대 학습 시간 (분)
    'patience': 100,                  # 개선 없을 시 조기 종료 에포크 수 (early stopping)

    # 최적화 설정
    'optimizer': 'auto',            # 옵티마이저 종류: 'SGD', 'Adam', 'AdamW'
    'lr0': 0.01,                   # 초기 학습률 (Adam 기준 추천: 1e-3 ~ 5e-4)
    'lrf': 0.01,                     # 최종 학습률 비율 (lr0 * lrf)
    'momentum': 0.937,                 # SGD 전용 모멘텀 계수
    'weight_decay': 0.0005,          # L2 정규화 계수 (overfitting 방지)
    'dropout': 0.0,                  # 드롭아웃 비율 (추천: 0.0~0.3)
    'freeze': [0],                   # 고정할 레이어 인덱스 또는 개수
    'nbs': 64,                       # 손실 정규화용 공칭 배치 크기
    'cos_lr': False,                 # 코사인 학습률 스케줄 사용 여부

    # 평가 및 손실 관련 설정
    'iou': 0.7,                     # NMS IoU 임계값 (추천: 0.5~0.7)
    'conf': 0.25,                    # confidence threshold (추천: 0.001~0.25)
    'box': 7.5,                     # 바운딩 박스 손실 가중치
    'cls': 0.5,                      # 클래스 손실 가중치 (클래스 불균형 시 ↑)
    'dfl': 1.5,                      # 분포 초점 손실 가중치 (RT-DETR에는 비활성화 가능)
    #'giou_loss_weight': 0.05,        # GIoU 손실 가중치
    #'cls_loss_weight': 0.05,         # 클래스 손실 가중치
    #'l1_loss_weight': 0.05,          # L1 위치 손실 가중치
    'warmup_epochs': 3.0,              # 워밍업 에포크 수 (추천: 2~5)
    'warmup_momentum': 0.8,          # 워밍업 시작 시 모멘텀
    'warmup_bias_lr': 0.1,           # 워밍업 시 bias에 대한 학습률

    # 데이터 증강 관련 설정
    'hsv_h': 0.015,                  # 색조 증강 범위 (추천: 0.0~0.05) -> 0.0 - 1.0
    'hsv_s': 0.7,                    # 채도 증강 범위 (추천: 0.0~0.9) -> 0.0 - 1.0
    'hsv_v': 0.4,                    # 명도 증강 범위 (추천: 0.0~0.9) -> 0.0 - 1.0
    'translate': 0.1,                # 평행이동 비율 (추천: 0.0~0.3) -> 0.0 - 1.0
    'scale': 0.5,                    # 스케일링 비율 (추천: 0.0~0.9) -> >=0.0
    'shear': 0.0,                    # 기울이기 비율 (추천: 0.0~0.2) -> -180 - +180
    'degrees': 0.0,                  # 회전 각도 (추천: 0~30) -> 0.0 - 180
    'perspective': 0.0,              # 원근 왜곡 정도 (추천: 0.0~0.001) -> 0.0 - 0.001
    'flipud': 0.0,                   # 상하 반전 확률 (추천: 0.0~0.5) -> 0.0 - 1.0
    'fliplr': 0.5,                   # 좌우 반전 확률 (추천: 0.0~0.5) -> 0.0 - 1.0
    'bgr': 0.0,                      # RGB ↔ BGR 전환 확률 (거의 사용되지 않음) -> 0.0 - 1.0
    'mosaic': 1.0,                   # Mosaic 증강 확률 (추천: 0.5~1.0) -> 0.0 - 1.0
    'mixup': 0.0,                    # MixUp 증강 확률 (추천: 0.0~0.5) -> 0.0 - 1.0
    'copy_paste': 0.0,               # Copy-Paste 증강 확률 (segmentation 전용) -> 0.0 - 1.0
    'erasing': 0.4,                  # Random Erasing 확률 (classification 전용) -> 0.0 - 0.9

    # 저장 및 시각화 설정
    'save': True,                    # 주석이 달린 이미지나 동영상을 파일로 저장
    'save_period': -1,               # 모델 저장 주기 (epoch 단위)
    'val': True,                     # 검증 데이터셋 평가 여부
    'plots': False,                  # 학습 곡선 및 시각화 플롯 저장
    'project': 'runs/rtdetr',        # 저장될 프로젝트 폴더
    'name': 'defect_detector',       # 실험 이름
    'save_dir': 'runs/rtdetr/custom', # 커스텀 저장 디렉토리
    'profile': False,                # ONNX, TensorRT 프로파일링 여부
    'visualize': False,               # 학습 중 시각화 여부

    # 포즈 및 마스크 설정
    'pose': 12.0,                   # 포즈 추정용 학습 여부
    'kobj': 2.0,                     # 포즈 객체성 손실 가중치
    'overlap_mask': True,           # 마스크 중첩 허용 여부
    'mask_ratio': 4,               # 마스크 다운샘플 비율 (0.25, 0.5, 1.0 등)
}