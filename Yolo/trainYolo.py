import wandb
import os
import random
import numpy as np
import torch
from ultralytics import YOLO

# 고정 설정
BASE_DIR = "Your/base/dir" # 기본 경로
DATA_PATH = os.path.join(BASE_DIR, "data.yaml") # 데이터셋 구성 파일
MODEL_PATH = "Use/YoloModel+(.pt)" # 사용할 yolo 모델
PROJECT_NAME = "Your Project Name"  # W&B 프로젝트 이름

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 에폭마다 정확도 기준으로 콘솔 + W&B 기록
def on_fit_epoch_end(trainer):
    epoch = trainer.epoch + 1

    try:
        if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
            # DetMetrics 객체
            metrics_obj = trainer.validator.metrics
            box_metrics = metrics_obj.box  # 핵심 평가값

            print(f"[Epoch {epoch:03}] mAP50={box_metrics.map50:.4f}, mAP50-95={box_metrics.map:.4f}, Precision={box_metrics.mp:.4f}, Recall={box_metrics.mr:.4f}")

            wandb.log({
                'val/mAP50': box_metrics.map50,
                'val/mAP5095': box_metrics.map,
                'val/Precision': box_metrics.mp,
                'val/Recall': box_metrics.mr,
                'epoch': epoch
            })
        else:
            print(f"[Epoch {epoch:03}] No validator metrics found.")
    except Exception as e:
        print(f"[Epoch {epoch:03}] Metric logging failed: {e}")

# 1. Sweep 설정
sweep_config = {
    'name': PROJECT_NAME,
    'method': 'bayes',  # 베이지안 최적화 방식
    'metric': {'name': 'metrics/mAP50', 'goal': 'maximize'},
    'parameters': {
        'imgsz': {'values': [1024, 1152, 1280]},  # 입력 이미지 크기 (고해상도)
        'lr0': {'min': 0.0001, 'max': 0.001},  # 초기 학습률
        'lrf': {'min': 0.05, 'max': 0.3},  # 최종 학습률
        #'warmup_epochs': {'values': [5]},  # Warmup epoch 수
        #'momentum': {'min': 0.7, 'max': 0.95},  # 모멘텀
        'weight_decay': {'min': 0.005, 'max': 0.05},  # L2 정규화
        'optimizer': {'values': ['AdamW']},  # 옵티마이저 종류 -> 'SGD' 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
        'dropout': {'min': 0.1, 'max': 0.4},  # 모델 내부 드롭아웃 확률 (과적합 방지)
        'hsv_h': {'min': 0.0, 'max': 0.05},  # 색상 증강(Hue 범위)
        'hsv_s': {'min': 0.4, 'max': 0.6},  # 채도 증강(Saturation)
        'hsv_v': {'min': 0.2, 'max': 0.6},  # 명도 증강(Value)
        'translate': {'min': 0.05, 'max': 1.0},  # 위치 이동 비율
        'scale': {'min': 0.4, 'max': 0.9},  # 크기 조정 비율
        'degrees': {'min': -10, 'max': 10},  # 회전 각도
        'mosaic': {'values': [0.0]},  # Mosaic 증강 비율
        'mixup': {'values': [0.0]},  # MixUp 증강 비율
        'fliplr': {'min': 0.0, 'max': 0.5},  # 좌우 반전 확률
        'flipud': {'min': 0.0, 'max': 0.5},  # 상하 반전 확률
    }
}

sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)

def sweep_train():
    with wandb.init():
        config = wandb.config
        
        run_name = f"hankum_{wandb.run.name}"
        save_dir = os.path.join(BASE_DIR, 'runs', PROJECT_NAME, run_name)
	
        # sweep config를 기반으로 학습 인자 구성
        train_args = {
        'model': MODEL_PATH,  # 사전학습된 모델 (yolov8m.pt)
        'data': DATA_PATH,  # 데이터셋 구성 파일(data.yaml)
        'imgsz': config.imgsz,
        'epochs': 200,  # 최대 학습 epoch 수
        #'warmup_epochs': config.warmup_epochs.
        'batch': 4,  # 배치 사이즈 (GPU 메모리에 맞춰 설정)
        
        # 옵티마이저 및 손실 관련
        'lr0': config.lr0,
        'lrf': config.lrf,
        #'momentum': config.momentum,
        'weight_decay': config.weight_decay,
        'optimizer': config.optimizer,
        'dropout': config.dropout,
        
        # Albumentations / Mosaic 관련 증강 파라미터
        'hsv_h': config.hsv_h,
        'hsv_s': config.hsv_s,
        'hsv_v': config.hsv_v,
        'translate': config.translate,
        'scale': config.scale,
        'degrees': config.degrees,
        'mosaic': config.mosaic,
        'mixup': config.mixup,
        'fliplr': config.fliplr,
        'flipud': config.flipud,
        
        # 학습 관련 추가 설정
        'amp': True,  # 자동 혼합 정밀도(Amp) 사용
        'val': True,  # validation 수행 여부
        'plots': False,  # 플롯 저장 여부
        'verbose': True,  # 로그 출력 여부
        'save': True,  # 모델 저장 여부
        'save_period': -1,  # 몇 에폭마다 저장할지 (-1이면 마지막만)
        'project': os.path.join(BASE_DIR, 'runs', PROJECT_NAME),  # 프로젝트 디렉토리
        'name': run_name,  # 실험 이름
        'exist_ok': True,  # 디렉토리 이미 존재해도 에러 없이 계속
        'device': 0,  # GPU 장치 번호 (0번 GPU 사용)
        'patience': 30,  # 조기 종료 조건 (개선되지 않는 epoch 수)
        
        # 그외
        'single_cls': True,  # 클래스 수를 1로 고정 (이진 분류처럼)
        'cos_lr': True,  # cosine learning rate scheduler 사용
        'iou': 0.6,  # 평가 시 IoU threshold
        'conf': 0.25  # confidence threshold
        #'close_mosaic': 10  # 몇 epoch 후 mosaic을 끌지 설정
        }

         # W&B에 전체 파라미터 기록
        wandb.config.update(train_args)
         # 모델 학습
        model = YOLO(train_args['model'])
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        wandb.watch(model)
        model.train(**train_args)
        # 검증 실행
        #val_results = model.val(data='/media/fourind/hdd/home/tmp/dataset/data.yaml', split='test', imgsz= config.imgsz, batch=2, iou=0.65)

        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            val_results = best_model.val(
                data=DATA_PATH,
                split='test',
                imgsz=config.imgsz,
                batch=2,
                iou=0.65
            )

        # 전체 평균 메트릭 로깅
        metrics = val_results.box
        # 정확도 등 검증 지표 로그
        wandb.log({
            'metrics/mAP50': metrics.map50, # mAP@0.5
            'metrics/mAP5095': metrics.map, # mAP@0.5:0.95
            'metrics/Precision': metrics.mp, # Precision
            'metrics/Recall': metrics.mr, # Recall
            'metrics/fitness': float(val_results.results_dict["fitness"]), # 종합 성능 지표
            'metrics/class': metrics.nc # 전체 클래스 수
        })

        # best.pt 파일 W&B Artifact로 저장
        # best.pt 저장
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Best yolov5m6u model for run {wandb.run.name}"
            )
            artifact.add_file(best_model_path, name="best.pt")
            wandb.log_artifact(artifact)
            print(f"[✓] Best model saved to WandB as artifact: {artifact.name}")
        else:
            print(f"[!] Best model file not found at {best_model_path}")

        # 최종 요약 출력
        print(f"[✓] mAP50: {metrics.map50:.4f}, mAP50-95: {metrics.map:.4f}, Precision: {metrics.mp:.4f}, Recall: {metrics.mr:.4f}, fitness: ", float(val_results.results_dict["fitness"]))
        # wandb.finish()  # 학습 완료 후 wandb 종료
        wandb.finish(quiet=True)

wandb.agent(sweep_id, function=sweep_train, count=0)  # count: 실행할 sweep 횟수
