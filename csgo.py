import torch
import pyautogui
import pynput
import time
from pynput.mouse import Button, Controller
from PIL import Image
import mss
from ctypes import CDLL
import numpy as np
import random
import yaml
def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates or loads a YOLOv5 model

    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 model
    """
    from pathlib import Path

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. '
                                       'You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).')
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. '
                                       'You will not be able to run inference with this model.')
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-nano model https://github.com/ultralytics/yolov5
    return _create('yolov5n', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create('yolov5s', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create('yolov5m', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create('yolov5l', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create('yolov5x', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-nano-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5n6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-small-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5s6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-medium-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5m6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-large-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5l6', pretrained, channels, classes, autoshape, _verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    # YOLOv5-xlarge-P6 model https://github.com/ultralytics/yolov5
    return _create('yolov5x6', pretrained, channels, classes, autoshape, _verbose, device)
# 初始化
gm = CDLL('ghub_device.dll')
gm.device_open()
with open('csgo.yaml', 'r') as file:
    config = yaml.safe_load(file)
model = custom(path=config['model_name'])  # custom
#model = torch.hub.load('', 'custom', 'csgo_5n.engine', source='local')
model.conf = config['conf']
press = False
mouse = Controller()
pyautogui.FAILSAFE = False
sct = mss.mss()
wide_x = config['wide_x']
wide_y = config['wide_y']
monitor = config['monitor']
pid_parameters = config['pid_parameters']
close = True


# 调用函数
def key_down(key=''):
    gm.key_down(key.encode('utf-8'))


def key_up(key=''):
    gm.key_up(key.encode('utf-8'))


def mouse_move(x, y, abs_move=False):
    gm.moveR(x, y, abs_move)


# int only
def mouse_down(key):
    gm.mouse_down(key)


def mouse_up(key):
    gm.mouse_up(key)


class PID(object):
    def __init__(self, kp, ki, kd, imax):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.imax = imax
        self.p = 0
        self.i = 0
        self.d = 0
        self.last = 0  # 上一次的误差值

    def cmd_pid(self, err, dt):
        # err代表准心到敌人坐标的相对移动值
        # dt代表程序循环一次所需要的时间
        self.p = err * self.Kp
        self.i = self.i + err * self.Ki * dt
        if self.i > self.imax:
            self.i = self.imax
        elif self.i < -self.imax:
            self.i = -self.imax
        self.d = (err - self.last) * self.Kd / dt
        self.last = err
        return self.p + self.i + self.d


def inference(model, img, size):
    global random_array
    results = model(img, size=size)
    # Convert PyTorch tensor to numpy array
    arr = results.xyxy[0].cpu().numpy()
    # Filter classes ending with 1 or 3
    class1_arr = arr[(arr[:, -1] == 1) | (arr[:, -1] == 3)]
    return class1_arr


def autoaim(df_head):
    global press, a, b, kp, ki, imax, pid_x, pid_y
    if press:
        point_x = wide_x/2
        point_y = wide_y/2
        # 求每个目标矩形中心点和点（point_x, point_y）之间的欧几里得距离
        distances = np.sqrt(((df_head[:, 0] + df_head[:, 2]) / 2 + ((wide_x/2)-160) - point_x) ** 2 + (
                (df_head[:, 1] + df_head[:, 3]) / 2 + ((wide_y/2)-160) - point_y) ** 2)
        # 返回距离最小的目标矩形
        min_distance_index = np.argmin(distances)
        min_distance_rect = df_head[min_distance_index]
        # 求每个目标矩形中心点和点（point_x, point_y）之间的欧几里得距离
        aim_x = int((min_distance_rect[0] + min_distance_rect[2]) / 2 + ((wide_x/2)-160))
        aim_y = int((min_distance_rect[1] + min_distance_rect[3]) / 2 + ((wide_y/2)-160))
        x = aim_x - wide_x / 2
        y = aim_y - wide_y / 2
        dt = time.time() - start_time
        move_x = pid_x.cmd_pid(x, dt)
        move_y = pid_x.cmd_pid(y, dt)
        a = max(a-b, 1)
        dx = 15*a
        dy = 15*a
        #print(move_x)
        if move_x >= dx:
            move_x = dx
        if move_x <= -dx:
            move_x = -dx
        if move_y >= dy:
            move_y = dy
        if move_y <= -dy:
            move_y = -dy
        mouse_move(int(move_x), int(move_y))


def create_number():
    if random.random() < 1:
        return 3
    else:
        return 0


def on_click1(x, y, button, pressed):
    global press, a, b, kp, ki, imax, pid_x, pid_y, close, random_array,kd
    if button == pynput.mouse.Button.x1 and pressed:
        press = True
        kp = pid_parameters['kp']
        ki = pid_parameters['ki']
        kd = pid_parameters['kd']
        imax = pid_parameters['imax']
        a = 4
        b = 0
        pid_x = PID(kp, ki, kd, imax)
        pid_y = PID(kp, ki, kd, imax)
        #random_array = create_number()
    if button == pynput.mouse.Button.x1 and not pressed:
        press = False
    if pressed and button == button.middle:
        random_array = create_number()
        close = not close
    if pressed and button == button.right:
        kp = pid_parameters['kp']
        ki = pid_parameters['ki']
        kd = pid_parameters['kd']
        imax = pid_parameters['imax']
        a = 4
        b = 0
        pid_x = PID(kp, ki, kd, imax)
        pid_y = PID(kp, ki, kd, imax)
        #random_array = create_number()
        press = True

def on_press(key):
    global press
    try:
        if key == pynput.keyboard.KeyCode.from_char('q'):
            press = False
    except AttributeError:
        print('Key pressed: {0}'.format(key))

if __name__ == '__main__':
    listener = pynput.mouse.Listener(on_click=on_click1)
    listener.start()
    keybord_listener = pynput.keyboard.Listener(on_press=on_press)
    keybord_listener.start()  # 非阻塞方式监控键盘
    while 1:
        if close:
            if press:
                start_time = time.time()
                df_head = np.empty((0, 6), float)
                screenshot = sct.grab(monitor)
                fullscreen = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                dataframe = inference(model, fullscreen, 320)
                if dataframe.shape[0] == 0:
                    pass
                else:
                    autoaim(dataframe)
                    end_time = time.time()
                    #print(1/(end_time-start_time))
            if not press:
                time.sleep(0.01)
        else:
            time.sleep(0.01)
