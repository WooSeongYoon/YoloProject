import os
import cv2
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, get_context
from functools import partial
import time

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc

# 전역 모델 및 설정 변수 초기화
global_model = None
global_params = {}

start_time = time.time()

# 워커 프로세스에서 YOLOX 모델 초기화 함수
def model_initializer(exp_file, ckpt_path, device, input_size, confthre, nmsthre, num_classes, patch_w, patch_h, root_folder):
    global global_model, global_params
    exp = get_exp(exp_file, None)
    model = exp.get_model().to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    global_model = model
    global_params = {
        "device": device,
        "input_size": input_size,
        "confthre": confthre,
        "nmsthre": nmsthre,
        "num_classes": num_classes,
        "patch_width": patch_w,
        "patch_height": patch_h,
        "root_folder": root_folder
    }

# 이미지 밝기 검사 (상하단 밝기 기준)
def check_brightness(img, pixel_height=100, threshold=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < pixel_height*2:
        return False
    top = gray[:pixel_height, :]
    bot = gray[-pixel_height:, :]
    return (int(top.mean()) < threshold and int(bot.mean()) < threshold)

# 입력 이미지를 8개의 패치로 분할
def split_patches(img, patch_width, patch_height):
    patches = []
    for iy in range(2):
        for ix in range(4):
            x0, y0 = ix * patch_width, iy * patch_height
            patch = img[y0:y0+patch_height, x0:x0+patch_width]
            patches.append(patch)
    return patches

# 탐지된 박스가 어두운 영역일 경우 오탐으로 판단
def false_positive_by_brightness(patch, box, image_height, thresh=5):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    positions = [(cx, 0), (cx, image_height - 1)]  # 이미지 최상단/최하단
    for px, py in positions:
        x0 = max(0, px - 15)
        y0 = max(0, py - 15)
        x1p = min(patch.shape[1], x0 + 30)
        y1p = min(patch.shape[0], y0 + 30)
        crop = patch[y0:y1p, x0:x1p]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return False  # 크롭 실패 → 오탐 아님
        if int(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) > thresh:
            return False  # 밝으면 오탐 아님
    return True  # 둘 다 어두우면 오탐

# COCO 박스를 YOLO 라벨 포맷으로 변환 (0~1 정규화)
def coco_box_to_yolo(box, patch_width, patch_height):
    """
    COCO box: [x1, y1, x2, y2, score, cls]
    YOLO box: class x_center y_center width height (모두 0~1)
    """
    x1, y1, x2, y2, score, cls = box
    x_center = (x1 + x2) / 2.0 / patch_width
    y_center = (y1 + y2) / 2.0 / patch_height
    w = (x2 - x1) / patch_width
    h = (y2 - y1) / patch_height
    return int(cls), x_center, y_center, w, h

# 단일 이미지 처리 함수
def process_one(img_path, result_dir, vision_dir, save_original, save_bbox_img, brightness_px, brightness_th):
    # 상대 경로 계산 및 하위 폴더 구조 유지
    rel_path = os.path.relpath(img_path, global_params["root_folder"])
    base = os.path.splitext(rel_path)[0]
    result_subdir = os.path.join(result_dir, os.path.dirname(base))
    vision_subdir = os.path.join(vision_dir, os.path.dirname(base))
    os.makedirs(result_subdir, exist_ok=True)
    if save_bbox_img:
        os.makedirs(vision_subdir, exist_ok=True)

    # 이미지 로드 및 밝기 필터링
    img = cv2.imread(img_path)
    if img is None or img.shape[0] < 1800 or img.shape[1] < 4096:
        print(f"이미지 불량: {img_path}")
        return
    if check_brightness(img, pixel_height=brightness_px, threshold=brightness_th):
        return

    # 상단 1800픽셀만 사용
    img_crop = img[:1800, :, :]
    patches = split_patches(img_crop, global_params["patch_width"], global_params["patch_height"])

    # 밝기 기준으로 패치 필터링 및 전처리
    imgs, ratios, valid_indices = [], [], []
    for i, patch in enumerate(patches):
        if check_brightness(patch, pixel_height=brightness_px, threshold=brightness_th):
            continue
        img_p, ratio = preproc(patch, global_params["input_size"])
        imgs.append(torch.from_numpy(img_p))
        ratios.append(ratio)
        valid_indices.append(i)

    if not imgs:
        return

    # 배치 추론 실행
    batch = torch.stack(imgs).float().to(global_params["device"])
    with torch.no_grad():
        outputs = global_model(batch)
        outputs = postprocess(outputs, global_params["num_classes"], global_params["confthre"], global_params["nmsthre"])

    # 결과 후처리 및 저장
    for idx, output in enumerate(outputs):
        i = valid_indices[idx]
        if output is None or output.shape[0] == 0:
            continue

        boxes = output.cpu().numpy()
        boxes_xyxy = boxes[:, :4] / ratios[idx]
        normal_labels = []
        false_positives = []

        for j, box in enumerate(boxes_xyxy):
            x1 = max(0, min(box[0], global_params["patch_width"]))
            y1 = max(0, min(box[1], global_params["patch_height"]))
            x2 = max(0, min(box[2], global_params["patch_width"]))
            y2 = max(0, min(box[3], global_params["patch_height"]))
            score = boxes[j, 4]
            cls = boxes[j, 5]
            if false_positive_by_brightness(patches[i], box, global_params["patch_height"]):
                false_positives.append([x1, y1, x2, y2, score, cls])
            else:
                normal_labels.append([x1, y1, x2, y2, score, cls])

        # 결과 저장 (YOLO 포맷 txt)
        label_path = os.path.join(result_subdir, f"{os.path.basename(base)}_patch{i+1}.txt")
        image_path = os.path.join(result_subdir, f"{os.path.basename(base)}_patch{i+1}.jpg")
        vision_path = os.path.join(vision_subdir, f"{os.path.basename(base)}_patch{i+1}_bbox.jpg")

        if normal_labels:
            with open(label_path, 'w') as f:
                for box in normal_labels:
                    yolo_label = coco_box_to_yolo(box, global_params["patch_width"], global_params["patch_height"])
                    _, xc, yc, w, h = yolo_label
                    # 0~1 범위만 저장
                    if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        f.write(f"{yolo_label[0]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            if save_original:
                cv2.imwrite(image_path, patches[i])

        # 바운딩 박스 시각화 저장
        if save_bbox_img:
            vis = patches[i].copy()
            for x1, y1, x2, y2, score, cls in normal_labels:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for x1, y1, x2, y2, score, cls in false_positives:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(vision_path, vis)

# 메인 실행 함수
def main(exp_file, ckpt_path, device, result_dir, save_original, save_bbox_img,
         input_size, confthre, nmsthre, num_classes, root_folder, num_workers=2,
         brightness_px=100, brightness_th=5):

    result_dir = os.path.join(result_dir)
    vision_dir = os.path.join(os.path.dirname(result_dir), "vision")
    os.makedirs(result_dir, exist_ok=True)
    if save_bbox_img:
        os.makedirs(vision_dir, exist_ok=True)

    # 입력 이미지 리스트 수집
    img_list = []
    for root, _, files in os.walk(root_folder):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.png')):
                img_list.append(os.path.join(root, fname))

    print(f"총 이미지: {len(img_list)} (각각 8패치 추론, 워커:{num_workers})")

    ctx = get_context("spawn")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # 병렬 처리를 위한 partial 함수 정의
    process_partial = partial(
        process_one,
        result_dir=result_dir,
        vision_dir=vision_dir,
        save_original=save_original,
        save_bbox_img=save_bbox_img,
        brightness_px=brightness_px,
        brightness_th=brightness_th
    )

    # 멀티프로세싱 풀 실행
    with ctx.Pool(
        processes=num_workers,
        initializer=model_initializer,
        initargs=(exp_file, ckpt_path, device, input_size, confthre, nmsthre, num_classes, 1024, 900, root_folder)
    ) as pool:
        list(tqdm(pool.imap_unordered(process_partial, img_list), total=len(img_list)))

    end_time = time.time()
    print(f"실행 시간 (전체): {end_time - start_time:.2f}초")

    # 처리 시간 측정 및 출력
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    print(f"총 실행 시간: {elapsed:.2f}초, 평균 {len(img_list)/elapsed:.2f} FPS (원본 기준)")

# 실행부
default_if_main_block = "__main__"
if __name__ == default_if_main_block:
    exp_file = "./runs/yolox_m/yolox_m.py"
    ckpt_path = "./runs/yolox_m/best_ckpt.pth"
    device = 'cuda'
    result_dir = './vision'
    input_size = (640, 768)
    confthre = 0.8
    nmsthre = 0.5
    num_classes = 1
    save_original = True
    save_bbox_img = True

    main(
        exp_file, ckpt_path, device, result_dir, save_original, save_bbox_img,
        input_size, confthre, nmsthre, num_classes,
        root_folder='./test_img',
        num_workers=2,
        brightness_px=100,
        brightness_th=5
    )
