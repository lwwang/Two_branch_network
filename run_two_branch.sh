if [ "$1" = "--train" ]; then
    CUDA_VISIBLE_DEVICES=0 \
    python train_embedding_nn.py \
    --image_feat_path /path/to/image_feature_train.mat \
    --sent_feat_path /path/to/text_feature_train.mat \
    --save_dir /path/to/save_dir/two-branch-ckpt
fi

if [ "$1" = "--test" ]; then
    CUDA_VISIBLE_DEVICES=1 \
    python eval_embedding_nn.py \
    --image_feat_path /path/to/image_feature_test.mat \
    --sent_feat_path /path/to/text_feature_test.mat \
    --restore_path /path/to/save_dir/two-branch-ckpt-$2.meta
fi

if [ "$1" = "--val" ]; then
    CUDA_VISIBLE_DEVICES=1 \
    python eval_embedding_nn.py \
    --image_feat_path /path/to/image_feature_val.mat \
    --sent_feat_path /path/to/text_feature_val.mat \
    --restore_path /path/to/save_dir/
fi
