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

# Default Model
#  single_fnet_default_time1513233124
# python single_fnet.py \
#     --prefix=default \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=last \
#     --use-gc true

# Default model with structure
# python single_fnet.py \
#     --prefix=default_structure \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=last \
#     --use-gc true \
#     --use-reconstruction true

# Default but using L_2 norm to normalize the output
# python single_fnet.py \
#     --prefix=default_l2norm \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=norm \
#     --use-gc true

# Default model with structure and l2 norm to normalize
# python single_fnet.py \
#     --prefix=default_structure_l2norm \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=norm \
#     --use-gc true \
#     --use-reconstruction true

# Default but with max-abs normalization and abs norm
# python single_fnet.py \
#     --prefix=default_absnorm \
#     --batch-size=64 \
#     --epoches=20 \
#     --data-size=30000 \
#     --l1-weight=10 \
#     --l2-weight=1. \
#     --lr=0.001 \
#     --norm-method=abs \
#     --use-gc true

# Default but with structural prediction and abs norm
python single_fnet.py \
    --gpu=0 \
    --prefix=default_structure_absnorm \
    --batch-size=4 \
    --epoches=20 \
    --data-size=30000 \
    --l1-weight=10 \
    --l2-weight=1. \
    --lr=0.001 \
    --norm-method=abs \
    --use-gc true \
    --use-reconstruction true \
    --dataset=kitti
