EXP_FILE="yolox-m"
PTH_FILE="/media/fourind/hdd/home/tmp/dataset/runs/yolox_s_640_all/best_ckpt.pth"
PY_FILE="/media/fourind/hdd/home/tmp/dataset/runs/yolox_s_640_all/yolox_s.py"

python -m tools.eval \
	-n  ${EXP_FILE} \
	-c ${PTH_FILE} \
	-f ${PY_FILE} \
	-b 4 \
	-d 1 \
	--conf 0.8 \
	--nms 0.5 \
	--fp16 \
	--test
