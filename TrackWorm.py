from ultralytics import YOLO
import cv2
import numpy as np
import math

# 自定义worm类别的四个关键点的连接关系
skeleton_worm = [
    [1, 2],  # 头连接身体 1
    [2, 3],  # 身体 1 连接身体 2
    [3, 4]   # 身体 2 连接尾巴
]

# 自定义worm2类别的五个关键点的连接关系
skeleton_worm2 = [
    [1, 2],  # 头连接身体 1
    [2, 3],  # 身体 1 连接身体 2
    [3, 4],  # 身体 2 连接身体 3
    [4, 5]   # 身体 3 连接尾巴
]

# 加载自定义的 YOLOv8 姿态估计模型
model_path = 'best.pt'
model = YOLO(model_path)

# 打开视频文件
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output_video_with_speed.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 记录上一帧的关键点
prev_keypoints = None

frame_count = 0

while cap.isOpened():
    # 读取一帧视频
    success, frame = cap.read()

    if success:
        # 使用模型预测
        results = model(frame)
        if len(results) > 0:
            # 获取检测结果中的类别信息和关键点数据
            boxes = results[0].boxes.cpu().numpy()
            keypoints_batch = results[0].keypoints.data.cpu().numpy()

            # 假设worm类别id为0，worm2类别id为1，这里只处理worm类别
            target_class_id = 0

            for i, box in enumerate(boxes):
                if box.cls[0] == target_class_id:
                    current_keypoints = keypoints_batch[i]
                    # 选择要显示速度的关键点索引，这里选择第一个关键点
                    speed_point_index = 0

                    # 在这一帧中初始化速度
                    speed = 0

                    if prev_keypoints is not None and i < len(prev_keypoints):
                        # 如果有上一帧的关键点，计算速度
                        prev_person_keypoints = prev_keypoints[i]
                        prev_point = prev_person_keypoints[speed_point_index]
                        curr_point = current_keypoints[speed_point_index]
                        if np.all(prev_point != 0) and np.all(curr_point != 0):  # 检查点是否有效
                            # 计算两帧之间的欧几里得距离
                            distance = math.sqrt((curr_point[0] - prev_point[0]) ** 2 + (curr_point[1] - prev_point[1]) ** 2)
                            speed = distance * fps  # 速度，像素/秒

                    # 根据不同类别选择不同的骨架连接关系
                    if target_class_id == 0:
                        skeleton = skeleton_worm
                    else:
                        skeleton = skeleton_worm2

                    # 绘制骨架
                    for connection in skeleton:
                        start_index, end_index = connection[0] - 1, connection[1] - 1
                        start_point = tuple(current_keypoints[start_index][:2].astype(int))
                        end_point = tuple(current_keypoints[end_index][:2].astype(int))
                        if np.all(current_keypoints[start_index] != 0) and np.all(current_keypoints[end_index] != 0):
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 1)

                    # 绘制所有关键点
                    for point in current_keypoints:
                        if np.all(point != 0):
                            point = tuple(point[:2].astype(int))
                            cv2.circle(frame, point, 3, (0, 0, 255), -1)

                    # 绘制指定关键点的速度
                    if np.all(current_keypoints[speed_point_index] != 0):
                        point = tuple(current_keypoints[speed_point_index][:2].astype(int))
                        speed_text = f"{speed:.1f}px/s"
                        cv2.putText(frame, speed_text, (point[0] + 5, point[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 保存当前帧的关键点
            prev_keypoints = keypoints_batch

        # 修正镜像问题，水平翻转
        frame = cv2.flip(frame, 1)

        # 将处理后的帧写入输出视频
        out.write(frame)

        frame_count += 1
    else:
        break

# 释放资源
cap.release()
out.release()
print(f"处理完成，输出视频保存为 {output_video_path}")