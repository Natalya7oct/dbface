from __future__ import annotations

import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from openvino.inference_engine import IECore
import streamlit as st
import time

class BBox:
    def __init__(self, label, xyrb, score=0, landmark=None, rotate=False):
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = (
            ','.join([str(item[:2]) for item in self.landmark])
            if self.landmark is not None
            else 'empty'
        )
        return (
            f'(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, '
            + f'b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})'
        )

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy_cat_landmark(self):
        x, y = zip(*self.landmark)
        return x + y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]

    # return cx, cy, cx.diff, cy.diff
    def safe_scale_center_and_diff(self, scale, limit_x, limit_y):
        cx = clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1)
        cy = clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1)
        return [int(cx), int(cy), cx - int(cx), cy - int(cy)]

    def safe_scale_center(self, scale, limit_x, limit_y):
        cx = int(clip_value((self.x + self.r) * 0.5 * scale, limit_x - 1))
        cy = int(clip_value((self.y + self.b) * 0.5 * scale, limit_y - 1))
        return [cx, cy]

    def clip(self, width, height):
        self.x = clip_value(self.x, width - 1)
        self.y = clip_value(self.y, height - 1)
        self.r = clip_value(self.r, width - 1)
        self.b = clip_value(self.b, height - 1)
        return self

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def intv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)


def floatv(*value):
    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([float(item) for item in value])
    elif isinstance(value, list):
        return [float(item) for item in value]
    elif value is None:
        return 0
    else:
        return float(value)


def clip_value(value, high, low=0):
    return max(min(value, high), low)


def randrf(low, high):
    return random.uniform(0, 1) * (high - low) + low


def mkdirs_from_file_path(path):
    try:
        path = path.replace('\\', '/')
        p0 = path.rfind('/')
        if p0 != -1:
            path = path[:p0]

            if not os.path.exists(path):
                os.makedirs(path)

    except Exception as e:
        print(e)


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # # image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    # # return image[:,:,(2,1,0)]
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image


def imwrite(path, image):
    path = path.replace('\\', '/')
    mkdirs_from_file_path(path)

    suffix = path[path.rfind('.'):]
    ok, data = cv2.imencode(suffix, image)

    if ok:
        try:
            with open(path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            print(e)
    return False


class RandomColor:
    def __init__(self, num):
        self.class_mapper = {}
        self.build(num)

    def build(self, num):
        self.colors = []
        for i in range(num):
            c = (i / (num + 1) * 360, 0.9, 0.9)
            t = np.array(c, np.float32).reshape(1, 1, 3)
            t = (
                cv2.cvtColor(t, cv2.COLOR_HSV2BGR)
                * 255
            ).astype(np.uint8).reshape(3)
            self.colors.append(intv(tuple(t)))

        seed = 0xFF01002
        length = len(self.colors)
        for i in range(length):
            a = i
            seed = (((i << 3) + 3512301) ^ seed) & 0x0FFFFFFF
            b = seed % length
            x = self.colors[a]
            y = self.colors[b]
            self.colors[a] = y
            self.colors[b] = x

    def get_index(self, label):
        if isinstance(label, int):
            return label % len(self.colors)
        elif isinstance(label, str):
            if label not in self.class_mapper:
                self.class_mapper[label] = len(self.class_mapper)
            return self.class_mapper[label]
        else:
            raise Exception(
                f'label is not support type{type(label)}, must be str or int',
            )

    def __getitem__(self, label):
        return self.colors[self.get_index(label)]


_rand_color = None


def randcolor(label, num=32):
    global _rand_color

    if _rand_color is None:
        _rand_color = RandomColor(num)
    return _rand_color[label]


# (239, 121, 162)
def drawbbox(
    image, bbox, color=None, thickness=2, textcolor=(0, 0, 0), landmarkcolor=(0, 0, 255),
):
    if color is None:
        color = randcolor(bbox.label)

    # text = f"{bbox.label} {bbox.score:.2f}"
    text = f'{bbox.score:.2f}'
    x, y, r, b = intv(bbox.box)
    w = r - x + 1

    cv2.rectangle(image, (x, y, r - x + 1, b - y + 1), color, thickness, 16)

    border = thickness / 2
    pos = (x + 3, y - 5)
    cv2.rectangle(
        image, intv(
            x - border, y - 21,
            w + thickness, 21,
        ), color, -1, 16,
    )
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

    if bbox.haslandmark:
        for i in range(len(bbox.landmark)):
            x, y = bbox.landmark[i][:2]
            cv2.circle(image, intv(x, y), 3, landmarkcolor, -1, 16)


def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[: image.shape[0], : image.shape[1], :] = image
        return newImage
    else:
        return image


def log(v):
    if isinstance(v, tuple) or isinstance(v, list) or isinstance(v, np.ndarray):
        return [log(item) for item in v]

    base = np.exp(1)
    if abs(v) < base:
        return v / base

    if v > 0:
        return np.log(v)
    else:
        return -np.log(-v)


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.array([exp(item) for item in v], v.dtype)

    gate = 1
    base = np.exp(1)
    if abs(v) < gate:
        return v * base

    if v > 0:
        return np.exp(v)
    else:
        return -np.exp(-v)


def file_name_no_suffix(path):
    path = path.replace('\\', '/')

    p0 = path.rfind('/') + 1
    p1 = path.rfind('.')

    if p1 == -1:
        p1 = len(path)
    return path[p0:p1]


def file_name(path):
    path = path.replace('\\', '/')
    p0 = path.rfind('/') + 1
    return path[p0:]



def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(exec_net, input_blob, image, threshold=0.4, nms_iou=0.5):
    outputs = exec_net.infer(inputs={input_blob: image})
    # print('outputs:', outputs)
    # print('outputs[\'Sigmoid_526\'].shape:', outputs['Sigmoid_526'].shape)
    # print('outputs[\'Exp_527\'].shape:', outputs['Exp_527'].shape)
    # print('outputs[\'Conv_525\'].shape:', outputs['Conv_525'].shape)
    hm, box, landmark = outputs['Sigmoid_526'], outputs['Exp_527'], outputs['Conv_525']
    # hm, box, landmark = outputs['1028'], outputs['1029'], outputs['1027']

    # print('outputs:', outputs)
    # print('outputs[\'1028\'].shape:', outputs['1028'].shape)
    # print('outputs[\'1029\'].shape:', outputs['1029'].shape)
    # print('outputs[\'1027\'].shape:', outputs['1027'].shape)

    hm = torch.from_numpy(hm).clone()
    box = torch.from_numpy(box).clone()
    landmark = torch.from_numpy(landmark).clone()

    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = (
        (hm == hm_pool).float() *
        hm
    ).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 5
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:0], x5y5[0:]))
        objs.append(
            BBox(
                0, xyrb=xyrb, score=score, landmark=box_landmark,
            ),
        )
    return nms(objs, iou=nms_iou)


def camera_demo(video, model_xml):
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    exec_net = ie.load_network(network=net, device_name='CPU')

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ok, frame = cap.read()

    # mean = [0.408, 0.447, 0.47]
    # std = [0.289, 0.274, 0.278]

    with st.empty():
        while ok:
            img = cv2.resize(frame, (640, 480))
            img = img[np.newaxis, :, :, :]
            img = img.transpose((0, 3, 1, 2))
            objs = detect(exec_net, input_blob, img)
            for obj in objs:
                drawbbox(frame, obj)
            st.image(frame,channels = "BGR")
            #cv2.imshow('demo DBFace', frame)
            #key = cv2.waitKey(1) & 0xFF
            #if key == ord('q'):
            #    break
            time.sleep(0.01)
            ok, frame = cap.read()

    cap.release()
    #cv2.destroyAllWindows()

