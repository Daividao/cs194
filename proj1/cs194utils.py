import skimage.transform
import cv2
import numpy as np

def crop(im, top, bottom, left, right):
    return im[top:bottom, left:right]

def resize(im, height, width):
    return skimage.transform.resize(im, (height, width))

def rotate(im, angle, center=None):
    """
    rotate an image `angle` degrees counterclockwise about `center`
    :param im (h, w, c) array
    :param angle (float) angle in degrees
    :param center (optional) point in image coordinates to rotate about
    """
    h, w = im.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    return cv2.warpAffine(im, mat, (w, h))


def translate(im, tx, ty):
    """
    translate an image `tx` to the right, `ty` down
    :param image (h, w, c) array
    :param tx (float) pixels to translate right
    :param ty (float) pixels to translate down
    """
    h, w = im.shape[:2]
    mat = translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)
    return cv2.warpAffine(im, mat, (w, h))

def tile_image(im, rep_x, rep_y):
  """
  :param im (h, w, c) array
  :param rep_x (int) times to repeat image in x dimension
  :param rep_y (int) times to repeat image in y dimension
  """
  return np.tile(im, [rep_y, rep_x, 1]);

def pad_image(im, pad_l, pad_r, pad_t, pad_b):
  """
  :param im (h, w, c) array
  :param pad_l (int) left padding
  :param pad_r (int) right padding
  :param pad_t (int) top padding
  :param pad_b (int) bottom padding
  """
  return np.pad(im, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)));

def create_mask(im, bg, thresh):
    mask = (abs(im[:,:,0] - bg[0]) < thresh) & \
       (abs(im[:,:,1] - bg[1]) < thresh) & \
       (abs(im[:,:,2] - bg[2]) < thresh)
    return mask

def replace_background(im, old_bg, new_bg):
    mask = create_mask(im, old_bg, 0.05)
    masked_im = np.copy(im)
    masked_im[mask, 0] = new_bg[0]
    masked_im[mask, 1] = new_bg[1]
    masked_im[mask, 2] = new_bg[2]
    return masked_im

def move_image_right(im, pixel):
    return np.roll(im,pixel,axis=1)

def move_image_down(im, pixel):
    return np.roll(im, pixel, axis=0)

def set_ith_row(im, i, value):
    im[i,:] = value

def set_ith_col(im, i, value):
    im[:,i] = value