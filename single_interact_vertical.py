import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
import os
import sys
from segment_anything import sam_model_registry, SamPredictor
from tkinter import filedialog
from PIL import Image
import math
Image.MAX_IMAGE_PIXELS = None


start = time.time()
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 20 / 255, 30 / 255, 0.1])
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

# 绘制边界框
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

sys.path.append("..")

# 模型的使用
sam_checkpoint = "model_check/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# filedialog.askopenfilename()函数选择一个文件
# 将选择的文件的路径存储在path中
path = filedialog.askopenfilename(title="请选择文件", initialdir=".", filetypes=(("jpeg files","*.jpg"),("all files","*.*")))

# os.path.normpath(path)用于标准化路径
# os.path.basename()用于提取路径中的文件名部分
jpg_name = os.path.basename(os.path.normpath(path))  # 选择的文件夹的名字
# cv2.imdecode()函数读取选择的图像文件
# np.fromfile(path, dtype=np.uint8)将文件读取为一个NumPy数组
image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
# image = cv2.imread(dir_path + '//' + file) # 读取图像

# **********************互动选点********************** #

# 创建两个空列表
positive = []
negative = []

# 定义一个函数来响应鼠标点击事件
def onclick(event):
    # 打印鼠标的位置和按键
    print('you pressed', event.button, event.xdata, event.ydata)
    # 判断鼠标按键
    if event.button == 1:  # 左键
        color = 'red'
    elif event.button == 3:  # 右键
        color = 'blue'
    else: # 其他按键
        color = 'black'
    # 在图像上显示一个圆点和坐标
    plt.scatter(event.xdata, event.ydata, s=50, c=color, marker='o')
    plt.annotate(f'({event.xdata:.2f}, {event.ydata:.2f})', xy=(event.xdata, event.ydata), xytext=(event.xdata+10, event.ydata+10))
    if event.button == 1:  # 左键
        positive.append((event.xdata, event.ydata))
    if event.button == 3:  # 右键
        negative.append((event.xdata, event.ydata))
    # 刷新图像
    plt.draw()

# 加载图像
img = Image.open(path).convert('RGB')
img = np.asarray(img)
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.figure(figsize=(20,20))
plt.title('左键添加位置，右键排除位置')
plt.imshow(img)
plt.axis('off')
# 连接鼠标点击事件和函数
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
# 显示图形界面
plt.show()

predictor.set_image(img)

# 添加正标签
input_point = np.array(positive)
input_label = np.ones(len(input_point))
# 添加负标签
negative_point = np.array(negative)
negative_label = np.zeros(len(negative_point))
input_point = np.concatenate((input_point, negative_point), axis=0)
input_label = np.append(input_label, negative_label)


# ************************************************** #

# ************************************************** #
# ******************根据取点加载mask******************* #
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
mask_input = logits[np.argmax(scores), :, :]

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
plt.figure(figsize=(20,20))
plt.imshow(img)
show_mask(masks, plt.gca())
plt.axis('off')
plt.show()
print(masks.shape)
# ************************************************** #

# ************************************************** #
# *******************识别mask边缘位置****************** #
df = pd.DataFrame(columns=['行数', '左外径', '左内径', '左壁厚', '右外径', '右内径', '右壁厚', '内径', '外径'])
image_height, image_width = img.shape[:2]
print(image_width)
print(image_height)
remainder = math.floor(image_height / 200)
print(remainder)
row_list = [i * 200 for i in range(1, remainder+1)]
print(row_list)
pos_list = []
thickness_left = []
thickness_right = []
inner_diameter = []
outside_diameter = []
pos0_list = []

for row in row_list:
    for pos in range(1, masks.shape[2]-2):
        if(masks[0][row][pos] == True and masks[0][row][pos+1] == False) or (masks[0][row][pos] == False and masks[0][row][pos+1] == True):
            pos_list.append(pos)
            for point in pos_list:
                cv2.circle(img, (pos, row), 3, (255, 0, 0), 5)
    temp = {'行数': row, '左外径': pos_list[0], '左内径': pos_list[1], '左壁厚': pos_list[1] - pos_list[0],
                '右内径': pos_list[2], '右外径': pos_list[3], '右壁厚': pos_list[3]-pos_list[2],
                '内径': pos_list[2]-pos_list[1], '外径': pos_list[3]-pos_list[0]}
    thickness_left.append(pos_list[1] - pos_list[0])
    thickness_right.append(pos_list[3] - pos_list[2])
    inner_diameter.append(pos_list[2] - pos_list[1])
    outside_diameter.append(pos_list[3] - pos_list[0])
    df = pd.concat([df, pd.DataFrame(temp, index=[0])], axis=0, ignore_index=True)
    pos0_list.append(pos_list[0])  # y坐标
    pos_list = []

k, b = np.polyfit(row_list, pos0_list, 1)
print("直线方程为: y = {:.4f}x + {:.2f}".format(k, b))
theta = math.atan(k)
fw = math.cos(theta)
def mul_theta(x):
    return x * fw
thickness_left = list(map(mul_theta, thickness_left))
thickness_right = list(map(mul_theta, thickness_right))
inner_diameter = list(map(mul_theta, inner_diameter))
outside_diameter = list(map(mul_theta, outside_diameter))

thickness_left_mean = np.mean(thickness_left)
thickness_right_mean = np.mean(thickness_right)
inner_diameter_mean = np.mean(inner_diameter)
outside_diameter_mean = np.mean(outside_diameter)

thickness_left_std = np.std(thickness_left)
thickness_right_std = np.std(thickness_right)
inner_diameter_std = np.std(inner_diameter)
outside_diameter_std = np.std(outside_diameter)

thickness_left_percent = "%.2f%%" % (thickness_left_std / thickness_left_mean * 100)
thickness_right_percent = "%.2f%%" % (thickness_right_std / thickness_right_mean * 100)
inner_diameter_percent = "%.2f%%" % (inner_diameter_std / inner_diameter_mean * 100)
outside_diameter_percent = "%.2f%%" % (outside_diameter_std / outside_diameter_mean * 100)

df.loc[17] = ['均值', '', '', thickness_left_mean, '', '', thickness_right_mean, inner_diameter_mean,
              outside_diameter_mean]
df.loc[18] = ['标准差', '', '', thickness_left_std, '', '', thickness_right_std, inner_diameter_std,
              outside_diameter_std]
df.loc[19] = ['标准差百分比', '', '', thickness_left_percent, '', '', thickness_right_percent,
              inner_diameter_percent, outside_diameter_percent]

csv_name = jpg_name.replace('.jpg', '.csv')
df.to_csv('F:\ALiYunDownload\segment-anything-main_10.8\segment-anything-main\output\data_csv\single_test\\' + csv_name, sep=',', index=False, encoding='utf_8_sig')  # 【待添加：自动创建文件夹】

plt.figure(figsize=(20,20))
# plt.figure()
plt.imshow(img)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')

plt.savefig('F:\ALiYunDownload\segment-anything-main_10.8\segment-anything-main\output\data_csv\single_test'+jpg_name, dpi=400)  # 【待添加：自动创建文件夹】
plt.show()
# ************************************************** #

end = time.time()
print(f'the runing time is:{end-start} s')