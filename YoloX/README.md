# YoloX
YoloX 모델의 학습 기초

## 사전 요구사항
- git clone "https://github.com/Megvii-BaseDetection/YOLOX"
- pip3 install -v -e .  # or  python3 setup.py develop
- coco데이터셋 폴더
- 각 파일 및 폴더 위치 설정

## YOLO프로젝트 구조
YoloProjectBase/   
├── Dataset/                # 데이터셋   
│   ├─ annotations/                # 데이터셋 정의   
│   │   ├── instances_test.json    # 학습 라벨   
│   │   ├── instances_train.json   # 검증 라벨   
│   │   └── instances_val.json     # 테스트 라벨   
│   ├── train/              # 학습 이미지   
│   ├── val/                # 검증 이미지   
│   └── test/               # 테스트 이미지   
│   
├── yolox폴더/              # yolox 공식 파일   
├── yoloTrain.sh            # yolox 학습 실행 코드   
├── tools/train_custom.py   # yolox 학습 코드   
├── yoloEval.sh             # yolox 검증 실행 코드   
├── tools/eval.py           # yolox 검증 코드   
└── exps/example/custom/yolox.py   # yolox 관련 설정 파일(위치 이동 가능)   
