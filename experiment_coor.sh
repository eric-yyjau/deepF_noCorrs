#! /bin/bash


# parser.add_argument('--prefix', default="test", type=str,
#                     help="Prefix for naming the log/model.")
# parser.add_argument('--batch-size', default=64, type=int,
#                     help="Batch size.")
# parser.add_argument('--epoches', default=30, type=int,
#                     help="Number of epoches.")
# parser.add_argument('--data-size', default=30000, type=int,
#                     help="Dataset size as number of (img1, img2, fmat) tuple")
# parser.add_argument('--l1-weight', default=10., type=float,
#                     help="Weight for L1-loss")
# parser.add_argument('--l2-weight', default=1., type=float,
#                     help='Weight for L2-loss')
# parser.add_argument('--lr', default=0.001, type=float,
#                     help='Learning rate')
# parser.add_argument('--use-gc', default=True, type=bool,
#                     help='Use gradient clipping.')
# parser.add_argument('--norm-method', default='last', type=str,
#                     help='Normalization method: (norm)|(abs)|(last)')
# parser.add_argument('--resume', default=None, type=str,
#                     help='Resuming model')
# parser.add_argument('--use-coor', default=False, type=bool,
#                     help='Whether using coordinate as input')
# parser.add_argument('--use-pos-index', default=False, type=bool,
#                     help='Whether using pos index')
# parser.add_argument('--use-reconstruction', default=False, type=bool,
#                     help='Whether using reconstruction layer')

# Default Model with pos index, last normalization
# python single_fnet.py \
#     --prefix=default_posidx \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=last \
#     --use-gc true \
#     --use-pos-index true


# # Default Model with pos index and structural prediction, last normalization
# Nikola05
# python single_fnet.py \
#     --prefix=default_posidx_structure \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=last \
#     --use-gc true \
#     --use-pos-index true \
#     --use-reconstruction true


# # Default Model with pos index, L2 normalization
# Nikola05 GPU2
# python single_fnet.py \
#     --prefix=default_posidx_l2norm \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=norm \
#     --use-gc true \
#     --use-pos-index true


# # Default Model with pos index and structural prediction, L2 normalization
# Amazon
# python single_fnet.py \
#     --prefix=default_posidx_l2norm_structure \ # last update from nikola05
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=norm \
#     --use-gc true \
#     --use-pos-index true \
#     --use-reconstruction true


# Default Model with pos index, L2 normalization
# Nikola05 GPU2
python single_fnet.py \
    --gpu-id=0 \
    --prefix=default_posidx_asbnorm \
    --batch-size=4\
    --epoches=60000 \
    --data-size=30000 \
    --l1-weight=10 \
    --l2-weight=1. \
    --lr=0.001 \
    --norm-method=norm \
    --use-pos-index true \
    --use-gc true \
    --use-reconstruction true \
    --dataset=kitti;
    # --resume log/single_fnet_default_posidx_asbnorm_time1525703151/model-50.ckpt;
    # --resume=log/single_fnet_default_posidx_asbnorm_time1523258893/model-5999.ckpt;


# Default Model with pos index and structural prediction, L2 normalization
# python single_fnet.py \
#     --prefix=default_posidx_absnorm_structure \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=abs \
#     --use-gc true \
#     --use-pos-index true \
#     --use-reconstruction true
