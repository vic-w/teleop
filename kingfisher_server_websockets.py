import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, Response
import threading
import os
import kingfisher
import asyncio
import struct
import yaml
import time

# from embodychain.devices.camera.king_fisher import kingfisher, RESOLUTION
# from embodychain.services import (
#     KINGFISHER_VIDEO_NAME,
#     KINGFISHER_VIDEO_PORT,
#     KINGFISHER_VIDEO_URL,
# )
# from embodychain.database import video_dir
video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
KINGFISHER_VIDEO_URL = "0.0.0.0"
KINGFISHER_VIDEO_PORT = 8012
KINGFISHER_VIDEO_NAME = "kingfisher"
c=kingfisher.connect("192.168.158.188")
width,height=kingfisher.get_resolution()
RESOLUTION = (int(height / 4), int(width / 4))
kingfisher.SetAUTO_EXPOSURE()
print("Resolution: ", RESOLUTION)
calib_file=kingfisher.getCalibData()
print("Calibration file: ", calib_file)

# with open("./calib_6.yaml", "r", encoding="utf-8") as f:
    # data = yaml.load(stream=f, Loader=yaml.FullLoader)
data = yaml.safe_load(calib_file)
# 将yaml中的数据转为numpy数组
R_l_r = np.array(data['R_l_r'])
cam1_k = np.array(data['cam1_k'])
cam2_k = np.array(data['cam2_k'])
dist_1 = np.array(data['dist_1']).reshape(-1)
dist_2 = np.array(data['dist_2']).reshape(-1)
t_l_r = np.array(data['t_l_r'])

# quarter size
cam1_k = cam1_k / 4
cam1_k[2, 2] = 1
cam2_k = cam2_k / 4
cam2_k[2, 2] = 1

print("R_l_r: ", R_l_r)
print("cam1_k: ", cam1_k)
print("cam2_k: ", cam2_k)
print("dist_1: ", dist_1)
print("dist_2: ", dist_2)
print("t_l_r: ", t_l_r)
print("Calibration data loaded.")
# 计算极线矫正矩阵
# 使用 cv2.stereoRectify 找到极线矫正的变换矩阵
rectify_scale = 0  # 缩放比例：0 - 不缩放，1 - 缩放
R1, R2, P1, P2, Q, valid1, valid2 = cv2.stereoRectify(cam1_k, None, cam2_k, None, (RESOLUTION[1], RESOLUTION[0]), R_l_r, t_l_r, flags=0, alpha=rectify_scale)
print("P2", P2)

map1x, map1y = cv2.initUndistortRectifyMap(cam1_k, None, R1, P1, (RESOLUTION[1], RESOLUTION[0]), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cam2_k, None, R2, P2, (RESOLUTION[1], RESOLUTION[0]), cv2.CV_32FC1)

from datetime import datetime
import os

CACHED_LOCK = threading.RLock()
FRAME_BUFFER = np.zeros(RESOLUTION + (3,), dtype=np.uint8)
LEFT_FRAME_BUFFER = np.zeros(RESOLUTION + (3,), dtype=np.uint8)
RIGHT_FRAME_BUFFER = np.zeros(RESOLUTION + (3,), dtype=np.uint8)


FPS = 30
VIDEO_LEN = 200
TMP_VIDEO = lambda x: "{}/{}.mp4".format(video_dir, x)  # 生成视频路径


app = FastAPI()


def generate_frames(
    write_video: bool = False, from_video: bool = False, video_path: str = None
):
    global device, CACHED_LOCK, FRAME_BUFFER, LEFT_FRAME_BUFFER, RIGHT_FRAME_BUFFER, TMP_VIDEO, VIDEO_LEN

    if from_video:
        assert os.path.exists(video_path), video_path
        video_capture = cv2.VideoCapture(video_path)  # single video
        write_video = False

    if write_video:
        time_now = datetime.now()
        current_time = time_now.strftime("%H:%M:%S").replace(":", "_")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 用于mp4格式的生成
        videowriter = cv2.VideoWriter(
            TMP_VIDEO(current_time + "_left"),
            fourcc,
            FPS,
            (RESOLUTION[1], RESOLUTION[0]),
        )
        videowriter_right = cv2.VideoWriter(
            TMP_VIDEO(current_time + "_right"),
            fourcc,
            FPS,
            (RESOLUTION[1], RESOLUTION[0]),
        )
    # 读帧
    count = 0
    count_capture = 0
    release = False
    while True:
        if from_video:
            ret, color_image = video_capture.read()
            if color_image is None:
                print("Get into loop!")
                video_capture = cv2.VideoCapture(TMP_VIDEO)
                ret, color_image = video_capture.read()
        else:
            print("Capture image: ", count_capture)
            count_capture += 1
            left, right = kingfisher.captureQuarterSize()
            ret = True
            color_image = np.reshape(left, RESOLUTION + (3,))
            color_image_right = np.reshape(right, RESOLUTION + (3,))
            dual_image = np.hstack([color_image, color_image_right])

        if write_video and count <= VIDEO_LEN:
            #print("Video count: {}".format(count))
            videowriter.write(color_image)
            videowriter_right.write(color_image_right)
            count += 1
        if count > VIDEO_LEN and write_video and not release:
            videowriter.release()
            videowriter_right.release()
            #print("Video capture done: {}".format(current_time))
            release = True

        if not ret:
            continue
        with CACHED_LOCK:
            FRAME_BUFFER = dual_image
            LEFT_FRAME_BUFFER = color_image
            RIGHT_FRAME_BUFFER = color_image_right
        ret, buffer = cv2.imencode(".jpg", dual_image)  # 将帧转化为JPEG格式
        frame = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # 生成帧数据


@app.get("/capture_image")
def capture_image():
    global CACHED_LOCK, FRAME_BUFFER
    with CACHED_LOCK:
        ret, buffer = cv2.imencode(".jpg", FRAME_BUFFER)
        frame = buffer.tobytes()
        return Response(content=frame)


@app.get("/{}".format(KINGFISHER_VIDEO_NAME))
def video():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    global map1x, map1y, map2x, map2y
    print("WebSocket connection")
    await websocket.accept()
    print("WebSocket connection accepted")
    n_frame = 0
    while True:
        n_frame += 1

        #print("Sending frame", n_frame)
        # 读取视频帧
        left, right = kingfisher.captureQuarterSize()

        # rectify
        left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
        # cv2.imshow("left", left)
        # cv2.imshow("right", right)
        # cv2.waitKey(0)

        # 编码成 JPEG 图片
        color_image = np.reshape(left, RESOLUTION + (3,))
        color_image_right = np.reshape(right, RESOLUTION + (3,))
        shape = [240, 135]
        color_image = cv2.resize(color_image, shape)
        color_image_right = cv2.resize(color_image_right, shape)
        _, buffer = cv2.imencode('.jpg', color_image)
        left_image_data = buffer.tobytes()
        _, buffer = cv2.imencode('.jpg', color_image_right)
        right_image_data = buffer.tobytes()
        
        # 计算两张图片的长度
        left_length = len(left_image_data)
        right_length = len(right_image_data)

        # 创建包头，包含两张图片的长度（每个长度使用 4 个字节）
        header = struct.pack('!II', left_length, right_length)

        # 发送给客户端
        #print("Sending frame to client")
        await websocket.send_bytes(header + left_image_data + right_image_data)
        #print("Frame sent")
        # 稍作等待，避免占用过多 CPU 资源
        await asyncio.sleep(0.001)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=KINGFISHER_VIDEO_URL, port=KINGFISHER_VIDEO_PORT, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
