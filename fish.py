# -*- coding: utf-8 -*-
"""
Screen‑capture + red‑blob tracking + trajectory (last 20 frames)
Press:
    y – toggle "lost detection" on/off
    q or ESC – quit
"""

import cv2
import numpy as np
from collections import deque
import mss   # pip install mss
import time
import pyautogui
# --------------------------------------------------------------
# 参数设置
# --------------------------------------------------------------
# 红色的 HSV 区间（可自行微调）
LOW_RED1  = np.array([0, 180, 70])
HIGH_RED1 = np.array([10, 255, 255])
LOW_RED2  = np.array([170, 180, 70])
HIGH_RED2 = np.array([180, 255, 255])
HIGH_QTE   = np.array([10, 160, 230])
LOW_QTE  = np.array([0, 175, 250])
#242 167 8
MIN_AREA   = 5          # 过滤太小的噪声块（像素）
MAX_HISTORY = 20          # 保留多少帧的轨迹点
LOST_FRAMES = 10          # 判定为“消失”需要连续未检测到的帧数

# --------------------------------------------------------------
# 状态变量
# --------------------------------------------------------------
pts = deque(maxlen=MAX_HISTORY)   # 中心点历史队列
lost_counter = 0                  # 连续未检测到的帧计数
detect_lost = False               # 是否打开“消失判定”，默认关闭

# --------------------------------------------------------------
# 屏幕捕获准备
# --------------------------------------------------------------
sct = mss.mss()
#monitor = sct.monitors[1]          # 第 1 个监视器（全屏），若只想抓取子区域请自行修改 dict
monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}
print(f"Capture region: {monitor}")

# --------------------------------------------------------------
# 主循环
# --------------------------------------------------------------
while True:
    while True:
        # ---- 屏幕抓图 ----
        img = np.array(sct.grab(monitor))          # BGR 格式的 ndarray (h, w, 4)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ---- 红色阈值分割 ----
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, LOW_RED1, HIGH_RED1)
        mask2 = cv2.inRange(hsv, LOW_RED2, HIGH_RED2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # 去噪（可选，提升鲁棒性）
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # ---- 找轮廓，取最大红块的中心点 ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maskqte = cv2.inRange(frame, LOW_QTE, HIGH_QTE)
    
        pixel_countr = int(cv2.countNonZero(mask))
        pixel_count = int(cv2.countNonZero(maskqte))
        print("[DEBUG]RED PIXELS:", pixel_countr, ", QTE PIXELS:", pixel_count)
        center = None
        if contours:
            # 只关心面积足够大的块
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area > MIN_AREA:
                M = cv2.moments(biggest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)

                    # 绘制检测到的红块轮廓（绿色）
                    cv2.drawContours(frame, [biggest], -1, (0, 255, 0), 2)
                    # 绘制中心点（小圆点，蓝色）
                    cv2.circle(frame, center, 4, (255, 0, 0), -1)

        # ---- 更新历史队列 & 丢失计数 ----
        if center:
            pts.appendleft(center)   # 最近的放在左侧，方便绘制顺序
            lost_counter = 0         # 检测到后重置计数
        else:
            lost_counter += 1

        # ---- 绘制轨迹（最近 MAX_HISTORY 帧）----
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            # 粗细随历史衰减，越老线条越细
            thickness = int(np.sqrt(MAX_HISTORY / float(i + 1)) * 2)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # ---- 消失判定显示（可开关）----
        if detect_lost and lost_counter > LOST_FRAMES:
            cv2.putText(frame,
                        "FISH!",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255), 3,
                        cv2.LINE_AA)
            pyautogui.rightClick()
            break
        # ---- 显示画面 & 键盘事件处理 ----
        cv2.imshow("Red Blob Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:          # q / ESC 退出
            break
        elif key == ord('y'):                     # y 键切换消失判定开关
            detect_lost = not detect_lost
            print(f"[INFO] Detect‑lost set to {detect_lost}")
    print("[INFO]Red mark lost, fishing")
    time.sleep(1)
    #第二次抓取屏幕图像，判断是否出现QTE（未实现）
    print("[INFO]Detecting QTE presence.")
    img = np.array(sct.grab(monitor))          # BGR 格式的 ndarray (h, w, 4)
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maskqte = cv2.inRange(hsv, LOW_QTE, HIGH_QTE)
    
    
    pixel_count = int(cv2.countNonZero(maskqte))
    print("Color match", pixel_count,"px")
    if pixel_count > 20:
        print("[INFO]Detected QTE")
        while True:
            # xxxc stands for xxx for comparison
            imgc = np.array(sct.grab(monitor))          # BGR 格式的 ndarray (h, w, 4)
            framec = cv2.cvtColor(imgc, cv2.COLOR_BGRA2BGR)
            hsvc = cv2.cvtColor(framec, cv2.COLOR_BGR2HSV)
            maskqtec = cv2.inRange(hsvc, LOW_QTE, HIGH_QTE)
            pixel_countc = int(cv2.countNonZero(maskqtec))
            print("[INFO]Grabbing image")
            if abs(pixel_countc - pixel_count) > 10:
                print("[INFO]GOTCHA!!")
                pyautogui.rightClick()
                break
           
        cv2.putText(frame,"QTE!",(60, 60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 0, 255), 3,cv2.LINE_AA)   
    else:
        print("[INFO]QTE not detected")
# --------------------------------------------------------------
# 清理资源
# --------------------------------------------------------------
cv2.destroyAllWindows()
