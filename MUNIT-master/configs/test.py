# #coding:utf8
# import cv2
# import os 
# import os.path
# import numpy as np
# imgs = []
# dir="/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/MUNIT-master/datasets/dataset/trainA/"
# for root, _, filenames in sorted(os.walk(dir)):
#     i = 1
#     for filename in filenames:
#         i += 1
#         if filename.endswith(".jpg"):
#             print(filename)
#             img = cv2.imread(dir+filename,0)
#             print(type(img))
#             rows, cols = img.shape
#             M = cv2.getRotationMatrix2D((cols/2,rows/2),30*i,1)
#             dst = cv2.warpAffine(img, M, (cols,rows))
#             cv2.imwrite("/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/MUNIT-master/datasets/dataset/trainB/"+filename,dst)
import numpy as np
import torch
def homography_based_on_top_corners_x_shift(rand_h):
    p = np.array([[1., 1., -1, 0, 0, 0, -(-1. + rand_h[0]), -(-1. + rand_h[0]), -1. + rand_h[0]],
                  [0, 0, 0, 1., 1., -1., 1., 1., -1.],
                  [-1., -1., -1, 0, 0, 0, 1 + rand_h[1], 1 + rand_h[1], 1 + rand_h[1]],
                  [0, 0, 0, -1, -1, -1, 1, 1, 1],
                  [1, 0, -1, 0, 0, 0, 1, 0, -1],
                  [0, 0, 0, 1, 0, -1, 0, 0, 0],
                  [-1, 0, -1, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, -1, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    b = np.zeros((9, 1), dtype=np.float32)
    b[8, 0] = 1.
    h = np.dot(np.linalg.inv(p), b)
    return torch.from_numpy(h).view(3, 3).cuda(1)


rand_h =[0,1,2,1,1,0,0,0,0,1,1,1,1,1,1]
a = homography_based_on_top_corners_x_shift(rand_h)
print(a.size())

def homography_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of homography matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Tensor): input batch of homography matrices (:math:`N \times 3 \times 3`)
        size (torch.Size): the target output image size (:math:`N \times C \times H \times W`)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Tensor): output Tensor of size (:math:`N \times H \times W \times 2`)
    """
    # y, x = torch.meshgrid((torch.linspace(-b, b, np.int(size[-2]*a)), torch.linspace(-b, b, np.int(size[-1]*a))))
    # n = np.int(size[-2] * a) * np.int(size[-1] * a)
    # hxy = torch.ones(n, 3, dtype=torch.float)
    # hxy[:, 0] = x.contiguous().view(-1)
    # hxy[:, 1] = y.contiguous().view(-1)
    # out = hxy[None, ...].cuda().matmul(theta.transpose(1, 2))
    # # normalize
    # out = out[:, :, :2] / out[:, :, 2:]
    # return out.view(theta.shape[0], np.int(size[-2]*a), np.int(size[-1]*a), 2)
import torch

# a =[[1, 2], [3, 4]]
a = torch.randn(2, 3)
print(a)
print(torch.cat(a))