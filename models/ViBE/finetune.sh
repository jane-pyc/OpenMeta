export WORK_DIR="./ViBE"
export PRETRAINED_MODEL="$WORK_DIR/models/pre-trained"
export DATA_DIR="$WORK_DIR/examples/fine-tune"
export CACHE_DIR="$DATA_DIR/cached"
export TRAIN_FILE="$DATA_DIR/vfdb_train.csv"
export DEV_FILE="$DATA_DIR/vfdb_process.csv"
export OUTPUT_DIR="$WORK_DIR/models/my_BPDR.250bp"

src/vibe fine-tune \
    --gpus 6,7 \
    --pre-trained_model $PRETRAINED_MODEL \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --cache_dir $CACHE_DIR \
    --max_seq_length 504 \
    --num_workers 20 \
    --num_train_epochs 1 \
    --eval_steps 80 \
    --per_device_batch_size 32 \
    --warmup_ratio 0.25 \
    --learning_rate 3e-5