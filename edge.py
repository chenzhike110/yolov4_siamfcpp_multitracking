import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from numba import jit


# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

# 统计概率霍夫线变换
@jit(nopython=True)
def offside_dectet(image, direction, ofplayer_x, ofplayer_y, dfplayer):
    # 找最后一名防守球员
    if direction == 'left':
        dfplayer_x = dfplayer[np.argmin(dfplayer[:,0]),0]
        dfplayer_y = dfplayer[np.argmin(dfplayer[:, 0]),1]
    elif direction == 'right':
        dfplayer_x = dfplayer[np.argmax(dfplayer[:,0]),0]
        dfplayer_y = dfplayer[np.argmax(dfplayer[:, 0]),1]
    elif direction == 'up':
        dfplayer_x = dfplayer[np.argmin(dfplayer[:,1]),0]
        dfplayer_y = dfplayer[np.argmin(dfplayer[:, 1]),1]
    elif direction == 'down':
        dfplayer_x = dfplayer[np.argmax(dfplayer[:,1]),0]
        dfplayer_y = dfplayer[np.argmax(dfplayer[:, 1]),1]

    has_offside = 0
    th = 10  # 边缘检测后大于th的才算边界

    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # x方向梯度
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # y方向梯度
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 各0.5的权重将两个梯度叠加
    dst, edges = cv2.threshold(edges, th, 255, cv2.THRESH_BINARY)  # 大于th的赋值255（白色）
    # cv2.imshow('edge', edges)

    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 255, minLineLength=min(gray.shape[0], gray.shape[1]) - 100,
                            maxLineGap=20)
    angle = []  # 备选线的角度
    b = []  # 备选线 y=kx+b的b
    if lines is None:
        has_line = 0
    else:
        has_line = 1
        for i in tqdm(range(lenth(lines))):
            x1, y1, x2, y2 = lines[i]
            angle_per = math.atan((y2 - y1) / (x2 - x1))  # 角度
            if angle_per < -np.pi / 4:  # 将角度换到-pi/4 ~ 3pi/4
                angle_per = angle_per + np.pi
            angle.append(angle_per)
            b.append(x1 * (y2 - y1) / (x2 - x1) - y1)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画线

        angle = np.array(angle)
        b = np.array(b)
        angle_delete_vertical = angle[(angle>(np.pi+0.2)/2) | (angle<(np.pi-0.2)/2)]
        b_delete_vertical = b[(angle>(np.pi+0.2)/2) | (angle<(np.pi-0.2)/2)]
        angle_ave = np.mean(angle_delete_vertical)  # 角度平均值
        angle_diff = angle_delete_vertical - angle_ave # 与平均值的差
        b = b_delete_vertical[abs(angle_diff) < 0.2]  # 去除离群点
        angle = angle_delete_vertical[abs(angle_diff) < 0.2]  # 去除离群点
        if len(angle) == 0:
            has_line = 0
        else:
            k_unsort = np.tan(angle)  # 角度对应的k
            b = np.array(b)
            dis = abs(-k_unsort * dfplayer_x + dfplayer_y + b) / np.sqrt(1 + k_unsort * k_unsort)  # 防守球员到线的距离
            dis = list(dis)
            angle_final = angle[dis.index(min(dis))]  # 选择离防守球员最近的线
            if abs(angle_final) < 0.001:  # 处理奇异情况
                if angle_final < 0:
                    angle_final = -0.001
                else:
                    angle_final = 0.001
            elif abs(angle_final) > 1.56 and abs(angle_final) < 1.58:
                angle_final = 1.56 * angle_final / abs(angle_final)
            k = np.tan(angle_final)  # 最终的k
            # 画出越位线
            y1_draw = int(dfplayer_y - k * dfplayer_x)
            y2_draw = int(k * gray.shape[1] - k * dfplayer_x + dfplayer_y)
            cv2.line(image, (0, y1_draw), (gray.shape[1], y2_draw), (0, 255, 0), 1)
            # 画出防守球员和进攻球员
            cv2.circle(image, (dfplayer_x, dfplayer_y), 5, (255, 0, 0))
            cv2.circle(image, (ofplayer_x, ofplayer_y), 5, (255, 0, 0))

            # 越位判罚
            line_x = ofplayer_y - (dfplayer_y - k * dfplayer_x) / k
            line_y = k * ofplayer_x - k * dfplayer_x + dfplayer_y
            if direction == 'left':
                if line_x > ofplayer_x:
                    has_offside = 1
            elif direction == 'right':
                if line_x < ofplayer_x:
                    has_offside = 1
            elif direction == 'up':
                if line_y > ofplayer_y:
                    has_offside = 1
            elif direction == 'down':
                if line_y < ofplayer_y:
                    has_offside = 1
        cv2.imshow("line_detect_possible_demo", image)

        if has_line == 1:
            print('has_line')
            if has_offside == 1:
                print('越位')
            else:
                print('不越位')
        else:
            print("no_line")
    return has_line, has_offside


