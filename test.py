import csv
import time
import pynput
from ctypes import CDLL
from pynput import keyboard
gm = CDLL('D:\zimiao\ghub_device.dll')
gm.device_open()
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
f = csv.reader(open('./ammo_path/m4a1.csv', encoding='utf-8'))
ak_recoil = []
for i in f:
    ak_recoil.append(i)
ak_recoil[0][0] = '0'
ak_recoil1 = [[float(i) for i in x] for x in ak_recoil]
f = csv.reader(open('./ammo_path/ak47.csv', encoding='utf-8'))
ak_recoil = []
for i in f:
    ak_recoil.append(i)
ak_recoil[0][0] = '0'
ak_recoil2 = [[float(i) for i in x] for x in ak_recoil]
f = csv.reader(open('./ammo_path/bizon.csv', encoding='utf-8'))
ak_recoil = []
for i in f:
    ak_recoil.append(i)
ak_recoil[0][0] = '0'
ak_recoil3 = [[float(i) for i in x] for x in ak_recoil]
f = csv.reader(open('./ammo_path/p90.csv', encoding='utf-8'))
ak_recoil = []
for i in f:
    ak_recoil.append(i)
ak_recoil[0][0] = '0'
ak_recoil4 = [[float(i) for i in x] for x in ak_recoil]
f = csv.reader(open('./ammo_path/guanbi.csv', encoding='utf-8'))
ak_recoil = []
for i in f:
    ak_recoil.append(i)
ak_recoil[0][0] = '0'
ak_recoil5 = [[float(i) for i in x] for x in ak_recoil]
i = 0
a = 0
press = False
ak_recoil = ak_recoil5
b = 0
close = True
def on_click(x, y, button, pressed):
    global press, a, b, i, ak_recoil, close
    if pressed and button == button.x1:
        i = 0
        a = 0
        press = True
    if pressed and button == button.middle:
        close = not close

def on_press(key):
    global ak_recoil, b
    try:
        if key == pynput.keyboard.KeyCode.from_char('n'):
            ak_recoil = ak_recoil3
            b = 8
            print("已切换为野牛")
        if key == pynput.keyboard.KeyCode.from_char('k'):
            ak_recoil = ak_recoil2
            b = 5
            print("已切换为ak47")
        if key == pynput.keyboard.KeyCode.from_char('o'):
            ak_recoil = ak_recoil1
            b = 5
            print("已切换为m4a1")
        if key == pynput.keyboard.KeyCode.from_char('p'):
            ak_recoil = ak_recoil4
            b = 8
            print("已切换为p90")
        if key == pynput.keyboard.KeyCode.from_char('-'):
            ak_recoil = ak_recoil5
            b = 0
            print("关闭了压枪")
    except AttributeError:
        print('Key pressed: {0}'.format(key))

listener = pynput.mouse.Listener(on_click=on_click)
listener.start()
keybord_listener = pynput.keyboard.Listener(on_press=on_press)
keybord_listener.start()#非阻塞方式监控键盘
while True:
    if close:
        if press:
            if a <= 1:
                a = a + 0.2
                time.sleep(0.03)
            else:
                if i < b:
                    k = -1
                    mouse_move(int(ak_recoil[i][0] * k), int(ak_recoil[i][1] * k))
                    gm.mouse_down(1)
                    time.sleep(ak_recoil[i][2] /1000)
                    gm.mouse_up(1)
                    i = i + 1
                else:
                    press = not press
        if not press:
            time.sleep(0.0001)
    else:
        time.sleep(0.0001)
