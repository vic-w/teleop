import asyncio
import websockets
import ssl
import json
from show_pose import get_ryhand_qpos, send_oculus_pose
import numpy as np
from scipy.spatial.transform import Rotation as R

#logfp = open('pose.log', 'w')
def _quaternion_to_matrix(quaternion: list) -> np.ndarray:
        x, y, z, w = quaternion
        r = R.from_quat([x, y, z, w])
        matrix = r.as_dcm()
        matrix_4x4 = np.eye(4)
        matrix_4x4[:3, :3] = matrix
        return matrix_4x4

def _parse_json_data(json_dict: dict):
    # 提取头部数据并解析成4x4矩阵
    head_matrix = None
    head_position = json_dict["head"]["position"]
    head_quaternion = json_dict["head"]["quaternion"]
    if head_position is not None and head_quaternion is not None:
        head_matrix = _quaternion_to_matrix(
            [
                head_quaternion["x"],
                head_quaternion["y"],
                head_quaternion["z"],
                head_quaternion["w"],
            ]
        )
        head_matrix[:3, 3] = [
            head_position["x"],
            head_position["y"],
            head_position["z"],
        ]

    # 提取右手数据并解析成列表
    right_hand_data = json_dict.get("right", {})

    # 初始化列表
    right_hand_list = []

    # 遍历右手数据并解析
    for key, value in right_hand_data.items():
        position = value["position"]
        quaternion = value["quaternion"]
        right_hand_list.append(
            [
                position["x"],
                position["y"],
                position["z"],
                quaternion["x"],
                quaternion["y"],
                quaternion["z"],
                quaternion["w"],
            ]
        )

    # 将解析后的列表转换为 4x4 矩阵
    right_hand_matrices = []

    for item in right_hand_list:
        position = item[:3]
        quaternion = item[3:]
        matrix = _quaternion_to_matrix(quaternion)
        matrix[:3, 3] = position
        right_hand_matrices.append(matrix)

    # 提取左手数据并解析成列表
    left_hand_data = json_dict.get("left", {})

    # 初始化列表
    left_hand_list = []

    # 遍历左手数据并解析
    for key, value in left_hand_data.items():
        position = value["position"]
        quaternion = value["quaternion"]
        left_hand_list.append(
            [
                position["x"],
                position["y"],
                position["z"],
                quaternion["x"],
                quaternion["y"],
                quaternion["z"],
                quaternion["w"],
            ]
        )

    # 将解析后的列表转换为 4x4 矩阵
    left_hand_matrices = []

    for item in left_hand_list:
        position = item[:3]
        quaternion = item[3:]
        matrix = _quaternion_to_matrix(quaternion)
        matrix[:3, 3] = position
        left_hand_matrices.append(matrix)

    if len(left_hand_matrices) == 0:
        left_hand_matrices = np.eye(4)
    elif left_hand_matrices[0].shape == (4, 4):
        left_hand_matrices = left_hand_matrices[0]
    else:
        left_hand_matrices = left_hand_matrices

    if len(right_hand_matrices) == 0:
        right_hand_matrices = np.eye(4)
    elif right_hand_matrices[0].shape == (4, 4):
        right_hand_matrices = right_hand_matrices[0]
    else:
        right_hand_matrices = right_hand_matrices

    return (
        head_matrix,
        left_hand_matrices,
        right_hand_matrices,
    )


# 处理客户端连接的异步函数
async def handle_connection(websocket, path=None):
    #print(f"New connection from {path}")
    try:
        async for message in websocket:
            #print(f"Received: {message}")
            json_dict = json.loads(message)
            if "left" in json_dict and json_dict["left"]:
                get_ryhand_qpos(json_dict['left'], 'left')
            if "right" in json_dict and json_dict["right"]:
                get_ryhand_qpos(json_dict['right'], 'right')

            head_matrix, left_hand_matrices, right_hand_matrices = _parse_json_data(json_dict)
            send_oculus_pose(head_matrix, left_hand_matrices, right_hand_matrices)

            #logfp.write(f"{message}\n")
            # 在这里添加你处理消息的逻辑
            await websocket.send("Response")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

# 启动 WebSocket 服务器
async def start_server():
    # SSL 上下文配置
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

    # 启动 WebSocket 服务器，指定 SSL 上下文
    server = await websockets.serve(
        handle_connection, '0.0.0.0', 8764, ssl=ssl_context
    )

    print("WebSocket server started on wss://0.0.0.0:8764")
    await server.wait_closed()

# 运行服务器
asyncio.run(start_server())
