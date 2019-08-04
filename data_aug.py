# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:43:42 2019

@author: cfd_Liu
"""

import cv2
import numpy as np
import random
import os
def load_aug_img(path, batchList, imgSize, imgInfo_train, is_training=True):
    imgbatch = []
    imgInfo = {}
    for i, filename in enumerate(batchList):
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, 1)
        if is_training:
            Info = imgInfo_train[filename].copy()
            bbox, w, h = parse_imgInfo(Info)
            
            img, bbox, w, h = data_augmentation(img, bbox, w, h, imgSize)
            img = img.astype(np.uint8)
            Info = get_imgInfo(bbox, w, h)
            imgInfo[filename] = Info
        img = cv2.resize(img, (imgSize, imgSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:].astype(np.float32) / 255
        imgbatch.append(img)
    imgbatch = np.concatenate(imgbatch, axis=0)
    return imgbatch, imgInfo
def data_augmentation(imgSrc, bboxSrc, w, h, imgSize):
    img = random_brightness(imgSrc)
    
    img, bbox = random_flip(img, bboxSrc)
    
    
    
    if np.random.uniform(0, 1) > 0.5:
        img, bbox = random_expand(img, bbox, 3)
        
    h, w, _ = img.shape
    if np.random.uniform(0, 1) > 0.5:
        img, bbox = resize_with_bbox(img, bbox, imgSize, imgSize)
        
    h, w, _ = img.shape
    bbox, crop = random_crop_with_constraints(bbox, (w, h))
    x0, y0, w, h = crop
    img = img[y0: y0+h, x0: x0+w]
    
    if not val_bbox(bbox):
        img = imgSrc
        bbox = bboxSrc
    
    h, w = img.shape[:2]
    return img, bbox, w, h
def val_bbox(bbox):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    if np.min(w)<1 or np.min(h)<1:
        return False
    return True

def parse_imgInfo(Info):
    w, h = Info[-1]
    bbox = np.zeros([len(Info)-1, 5], np.int32)
    for i, obj in enumerate(Info[:-1]):
        bbox[i] = [obj[1]['xmin'], obj[1]['ymin'], obj[1]['xmax'], obj[1]['ymax'], obj[0]]
    return bbox, w, h
def get_imgInfo(bbox, w, h):
    Info = []
    for obj in bbox:
        tmp = [obj[4], {'xmin':obj[0], 'ymin':obj[1], 'xmax':obj[2], 'ymax':obj[3]}]
        Info.append(tmp)
    Info.append([w, h])
    return Info
def letterbox_resize(img, new_width, new_height):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h))
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh

def resize_with_bbox(img, bbox, new_width, new_height):
    '''
    Resize the image and correct the bbox accordingly.
    '''

    image_padded, resize_ratio, dw, dh = letterbox_resize(img, new_width, new_height)

    # xmin, xmax
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * resize_ratio + dw
    # ymin, ymax
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * resize_ratio + dh
    return image_padded, bbox

def random_flip(img, bbox, px=0.5, py=0):
    '''
    Randomly flip the image and correct the bbox.
    param:
    px:
        the probability of horizontal flip
    py:
        the probability of vertical flip
    '''
    bbox = bbox.copy()
    height, width = img.shape[:2]
    if np.random.uniform(0, 1) < px:
        img = cv2.flip(img, 1)
        xmax = width - bbox[:, 0]
        xmin = width - bbox[:, 2]
        bbox[:, 0] = xmin
        bbox[:, 2] = xmax

    if np.random.uniform(0, 1) < py:
        img = cv2.flip(img, 0)
        ymax = height - bbox[:, 1]
        ymin = height - bbox[:, 3]
        bbox[:, 1] = ymin
        bbox[:, 3] = ymax
    return img, bbox

def random_brightness(img, brightness_delta=32, p=0.5):
    if np.random.uniform(0, 1) > p:
        img = img.astype(np.float32)
        brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
        img = img + brightness_delta
    return np.clip(img, 0, 255)

def random_expand(img, bbox, max_ratio=4, fill=0, keep_ratio=True):
    '''
    Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    param:
    max_ratio :
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill :
        The value(s) for padded borders.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.
    '''
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=img.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = img

    # correct bbox
    bbox[:, :2] += (off_x, off_y)
    bbox[:, 2:4] += (off_x, off_y)

    return dst, bbox

def bbox_crop(bbox, crop_box=None, allow_outside_center=True):
    """Crop bounding boxes according to slice area.
    This method is mainly used with image cropping to ensure bonding boxes fit
    within the cropped image.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    crop_box : tuple
        Tuple of length 4. :math:`(x_{min}, y_{min}, width, height)`
    allow_outside_center : bool
        If `False`, remove bounding boxes which have centers outside cropping area.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape (M, 4+) where M <= N.
    """
    bbox = bbox.copy()
    if crop_box is None:
        return bbox
    if not len(crop_box) == 4:
        raise ValueError(
            "Invalid crop_box parameter, requires length 4, given {}".format(str(crop_box)))
    if sum([int(c is None) for c in crop_box]) == 4:
        return bbox

    l, t, w, h = crop_box

    left = l if l else 0
    top = t if t else 0
    right = left + (w if w else np.inf)
    bottom = top + (h if h else np.inf)
    crop_bbox = np.array((left, top, right, bottom))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(crop_bbox[:2] <= centers, centers < crop_bbox[2:]).all(axis=1)

    # transform borders
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bbox[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop_bbox[2:4])
    bbox[:, :2] -= crop_bbox[:2]
    bbox[:, 2:4] -= crop_bbox[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:4]).all(axis=1))
    bbox = bbox[mask]
    return bbox

def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def random_crop_with_constraints(bbox, size, min_scale=0.3, max_scale=1,
                                 max_aspect_ratio=2, constraints=None,
                                 max_trial=50):
    """Crop an image randomly with bounding box constraints.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Parameters
    ----------
    bbox : numpy.ndarray
        Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
        The second axis represents attributes of the bounding box.
        Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        we allow additional attributes other than coordinates, which stay intact
        during bounding box transformations.
    size : tuple
        Tuple of length 2 of image shape as (width, height).
    min_scale : float
        The minimum ratio between a cropped region and the original image.
        The default value is :obj:`0.3`.
    max_scale : float
        The maximum ratio between a cropped region and the original image.
        The default value is :obj:`1`.
    max_aspect_ratio : float
        The maximum aspect ratio of cropped region.
        The default value is :obj:`2`.
    constraints : iterable of tuples
        An iterable of constraints.
        Each constraint should be :obj:`(min_iou, max_iou)` format.
        If means no constraint if set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`.
        If this argument defaults to :obj:`None`, :obj:`((0.1, None), (0.3, None),
        (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
    max_trial : int
        Maximum number of trials for each constraint before exit no matter what.
    Returns
    -------
    numpy.ndarray
        Cropped bounding boxes with shape :obj:`(M, 4+)` where M <= N.
    tuple
        Tuple of length 4 as (x_offset, y_offset, new_width, new_height).
    """
    # default params in paper
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
#            (None, 1),
        )

    w, h = size

    candidates = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        min_iou = -np.inf if min_iou is None else min_iou
        max_iou = np.inf if max_iou is None else max_iou

        for _ in range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(h - crop_h)
            crop_l = random.randrange(w - crop_w)
            crop_bb = np.array((crop_l, crop_t, crop_l + crop_w, crop_t + crop_h))

            if len(bbox) == 0:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                return bbox, (left, top, right-left, bottom-top)

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                top, bottom = crop_t, crop_t + crop_h
                left, right = crop_l, crop_l + crop_w
                candidates.append((left, top, right-left, bottom-top))
                break

    # random select one
    while candidates:
        crop = candidates.pop(np.random.randint(0, len(candidates)))
        new_bbox = bbox_crop(bbox, crop, allow_outside_center=False)
        if new_bbox.size < 1:
            continue
        new_crop = (crop[0], crop[1], crop[2], crop[3])
        return new_bbox, new_crop
    return bbox, (0, 0, w, h)

if __name__ == '__main__':
    path_train = np.load('./data/path_train.npy').item()[1]
    imgSize_train = 416
    imgInfo_train = np.load('./data/dataset_train.npy').item()
    trainList = np.load('./data/trainList.npy').item()[1]
    batchList = trainList[:5]
    
    Info_tmp = {}
    for i in batchList:    
        Info_tmp[i] = imgInfo_train[i]
        
    imgbatch, imgInfo = load_aug_img(path_train, batchList, imgSize_train, imgInfo_train)
    
    for i, img in enumerate(imgbatch):
        img = np.uint8(img * 255)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        Info = imgInfo[batchList[i]]
        w, h = Info[-1]
        for obj in Info[:-1]:
            x1 = obj[1]['xmin'] / w *416; x2 = obj[1]['xmax'] / w *416; y1 = obj[1]['ymin'] / h *416; y2 = obj[1]['ymax'] / h *416;
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imshow('%d' %i, img)