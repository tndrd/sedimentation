import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import tqdm

PX2MM_COEFF = 50 / (108 - 40) # 108 pm 1, 50 pm 1

ROTATION = 2 # deg
CROP_X = [430, 515] # 540 to see ruler
CROP_Y = [510, 780]

def get_names(prefix, metastrat):
    query = os.path.join(*prefix, "*.png")
    names = glob.glob(query)

    names.sort(key=metastrat)
    metas = [metastrat(name) for name in names]

    return names, metas

def px2mm(img, px):
   return (img.shape[0] - px) * PX2MM_COEFF

def mm2px(img, mm):
   return img.shape[0] - mm / PX2MM_COEFF

def display_img(img, bgr2rgb=True):
    plt.figure(dpi=200)

    if bgr2rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = plt.imshow(img, interpolation="none")

    plt.tight_layout()

    ticks_mm = np.arange(0, px2mm(img, 0), 10)
    ticks_px = mm2px(img, ticks_mm)

    plt.xticks([])
    plt.yticks(ticks_px, labels=[str(round(mm)) for mm in ticks_mm])

    return ax

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def crop_image(image, x_bounds, y_bounds):
   return image[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]

def translate_img(img, delta):
   h, w, _ = img.shape
   T = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])

   return cv2.warpAffine(img, T, (w, h))


COLORFIT_K = 0.00426302
COLORFIT_B = 0.5425
COLORFIT_A = np.array([-0.62989275, -0.61689097, -0.47189051])
COLORFIT_R0 = np.array([65.66666667, 72.5, 87.])

def bgr2buoyancy(bgr):
   t = np.dot(COLORFIT_A.T, (bgr - COLORFIT_R0))
   return COLORFIT_K * t + COLORFIT_B