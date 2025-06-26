# 실험명, exp 파일, 기타 옵션은 원하는 대로 수정
EXP_FILE="exps/example/custom/yolox_m.py"
MODEL_NAME="./yolox_s"
BATCH_SIZE=8
PROJECT="my_yolox_project"

python -m yolox.tools.train_custom \
	-n ${MODEL_NAME} \
	-b ${BATCH_SIZE} \
	-d 1 \
	-f ${EXP_FILE} \
	--fp16 \
	--project ${PROJECT} \
	-o \
	--logger wandb wandb-project ${PROJECT}
