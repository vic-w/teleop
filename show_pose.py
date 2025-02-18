import cv2
import numpy as np
import json
from finger_rotation import get_relative_rotation
from publish_topic import hand_controller


# 连接手指关节编号，生成连接线
lines = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8, 9],
    [0, 10, 11, 12, 13, 14],
    [0, 15, 16, 17, 18, 19],
    [0, 20, 21, 22, 23, 24]
]

hands = hand_controller()

def get_unit_vector(x, y, z, qx, qy, qz, qw):
    R = np.array([[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                    [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                    [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]])
    x_end = (x + int(R[0][0] * 50), y + int(R[1][0] * 50), z + int(R[2][0] * 50))
    y_end = (x + int(R[0][1] * 50), y + int(R[1][1] * 50), z + int(R[2][1] * 50))
    z_end = (x + int(R[0][2] * 50), y + int(R[1][2] * 50), z + int(R[2][2] * 50))
    return x_end, y_end, z_end

def show_pose(json_dict):
    # 生成一副正视图，一副俯视图。正视图左右各1米，上下高2米。俯视图左右各1米，前后各1米。两张图都是400*400像素。左右拼接成一张800*400像素的图。
    # 正视图
    front_view = np.zeros((400, 400, 3), np.uint8)
    # 俯视图
    top_view = np.zeros((400, 400, 3), np.uint8)
    # 拼接图
    combined_view = np.zeros((400, 800, 3), np.uint8)
    # 分界线
    cv2.line(combined_view, (400, 0), (400, 400), (255, 255, 255), 2)

    # 获取头部位置
    head_position = json_dict["head"]["position"]   

    # 画出头部
    head_x = int((head_position["x"]) * 200)
    head_y = int((head_position["y"]) * 200)
    head_z = int((head_position["z"]) * 200)
    head_qx = json_dict["head"]["quaternion"]["x"]
    head_qy = json_dict["head"]["quaternion"]["y"]
    head_qz = json_dict["head"]["quaternion"]["z"]
    head_qw = json_dict["head"]["quaternion"]["w"]
    # 转换成旋转矩阵R
    R = np.array([[1 - 2 * head_qy * head_qy - 2 * head_qz * head_qz, 2 * head_qx * head_qy - 2 * head_qz * head_qw, 2 * head_qx * head_qz + 2 * head_qy * head_qw],
                    [2 * head_qx * head_qy + 2 * head_qz * head_qw, 1 - 2 * head_qx * head_qx - 2 * head_qz * head_qz, 2 * head_qy * head_qz - 2 * head_qx * head_qw],
                    [2 * head_qx * head_qz - 2 * head_qy * head_qw, 2 * head_qy * head_qz + 2 * head_qx * head_qw, 1 - 2 * head_qx * head_qx - 2 * head_qy * head_qy]])
    

    #print("head x:%f, y:%f, z:%f" % (head_position["x"], head_position["y"], head_position["z"]))
    #print("head x:%d, y:%d, z:%d" % (head_x, head_y, head_z))
    cv2.circle(front_view, (head_x + 200, 400 - head_y), 5, (0, 0, 255), -1)
    cv2.circle(top_view, (head_x + 200, 200 - head_z), 5, (0, 0, 255), -1)

    # 画出头部朝向
    # 计算出head的X轴的单位向量的末端坐标的（x，y，z）
    head_x_end = (head_x + int(R[0][0] * 50), head_y + int(R[1][0] * 50), head_z + int(R[2][0] * 50))
    head_y_end = (head_x + int(R[0][1] * 50), head_y + int(R[1][1] * 50), head_z + int(R[2][1] * 50))
    head_z_end = (head_x + int(R[0][2] * 50), head_y + int(R[1][2] * 50), head_z + int(R[2][2] * 50))



    # 画正视图x
    cv2.line(front_view, (head_x + 200, 400 - head_y), (head_x_end[0] + 200, 400 - head_x_end[1]), (0, 0, 255), 2)
    # 画正视图y
    cv2.line(front_view, (head_x + 200, 400 - head_y), (head_y_end[0] + 200, 400 - head_y_end[1]), (0, 255, 0), 2)
    # 画正视图z
    cv2.line(front_view, (head_x + 200, 400 - head_y), (head_z_end[0] + 200, 400 - head_z_end[1]), (255, 0, 0), 2)
    # 画俯视图x
    cv2.line(top_view, (head_x + 200, 200 - head_z), (head_x_end[0] + 200, 200 - head_x_end[2]), (0, 0, 255), 2)
    # 画俯视图y
    cv2.line(top_view, (head_x + 200, 200 - head_z), (head_y_end[0] + 200, 200 - head_y_end[2]), (0, 255, 0), 2)
    # 画俯视图z
    cv2.line(top_view, (head_x + 200, 200 - head_z), (head_z_end[0] + 200, 200 - head_z_end[2]), (255, 0, 0), 2)




    #

    # 如左手存在，获取左手所有点的位置
    if "left" in json_dict and json_dict["left"]:
        left_points = []
        for i in range(25):
            if str(i) in json_dict["left"]:
                left_position = json_dict["left"][str(i)]["position"]
                left_x = int((left_position["x"]) * 200)
                left_y = int((left_position["y"]) * 200)
                left_z = int((left_position["z"]) * 200)
                cv2.circle(front_view, (left_x + 200, 400 - left_y), 1, (255, 255, 0), -1)
                cv2.circle(top_view, (left_x + 200, 200 - left_z), 1, (255, 255, 0), -1)
                left_points.append([left_x, left_y, left_z])

        #print('left_points:', left_points[0])
        # 画出连接线
        for line in lines:
            for i in range(len(line) - 1):
                index1 = line[i]
                index2 = line[i + 1]
                cv2.line(front_view, (left_points[index1][0] + 200, 400 - left_points[index1][1]), (left_points[index2][0] + 200, 400 - left_points[index2][1]), (255, 255, 0), 2)
                cv2.line(top_view, (left_points[index1][0] + 200, 200 - left_points[index1][2]), (left_points[index2][0] + 200, 200 - left_points[index2][2]), (255, 255, 0), 2)
        
        get_ryhand_qpos(json_dict['left'])

        # 画手腕的方向
        wrist_x = left_points[0][0]
        wrist_y = left_points[0][1]
        wrist_z = left_points[0][2]
        wrist_qx = json_dict["left"]["0"]["quaternion"]["x"]
        wrist_qy = json_dict["left"]["0"]["quaternion"]["y"]
        wrist_qz = json_dict["left"]["0"]["quaternion"]["z"]
        wrist_qw = json_dict["left"]["0"]["quaternion"]["w"]
        wrist_x_end, wrist_y_end, wrist_z_end = get_unit_vector(wrist_x, wrist_y, wrist_z, wrist_qx, wrist_qy, wrist_qz, wrist_qw)
        # 画正视图x
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_x_end[0] + 200, 400 - wrist_x_end[1]), (0, 0, 255), 1)
        # 画正视图y
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_y_end[0] + 200, 400 - wrist_y_end[1]), (0, 255, 0), 1)
        # 画正视图z
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_z_end[0] + 200, 400 - wrist_z_end[1]), (255, 0, 0), 1)
        # 画俯视图x
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_x_end[0] + 200, 200 - wrist_x_end[2]), (0, 0, 255), 1)
        # 画俯视图y
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_y_end[0] + 200, 200 - wrist_y_end[2]), (0, 255, 0), 1)
        # 画俯视图z
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_z_end[0] + 200, 200 - wrist_z_end[2]), (255, 0, 0), 1)
        
        wrist_x = left_points[10][0]
        wrist_y = left_points[10][1]
        wrist_z = left_points[10][2]
        wrist_qx = json_dict["left"]["10"]["quaternion"]["x"]
        wrist_qy = json_dict["left"]["10"]["quaternion"]["y"]
        wrist_qz = json_dict["left"]["10"]["quaternion"]["z"]
        wrist_qw = json_dict["left"]["10"]["quaternion"]["w"]
        wrist_x_end, wrist_y_end, wrist_z_end = get_unit_vector(wrist_x, wrist_y, wrist_z, wrist_qx, wrist_qy, wrist_qz, wrist_qw)
        # 画正视图x
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_x_end[0] + 200, 400 - wrist_x_end[1]), (0, 0, 255), 1)
        # 画正视图y
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_y_end[0] + 200, 400 - wrist_y_end[1]), (0, 255, 0), 1)
        # 画正视图z
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_z_end[0] + 200, 400 - wrist_z_end[1]), (255, 0, 0), 1)
        # 画俯视图x
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_x_end[0] + 200, 200 - wrist_x_end[2]), (0, 0, 255), 1)
        # 画俯视图y
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_y_end[0] + 200, 200 - wrist_y_end[2]), (0, 255, 0), 1)
        # 画俯视图z
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_z_end[0] + 200, 200 - wrist_z_end[2]), (255, 0, 0), 1)
        
        wrist_x = left_points[14][0]
        wrist_y = left_points[14][1]
        wrist_z = left_points[14][2]
        wrist_qx = json_dict["left"]["14"]["quaternion"]["x"]
        wrist_qy = json_dict["left"]["14"]["quaternion"]["y"]
        wrist_qz = json_dict["left"]["14"]["quaternion"]["z"]
        wrist_qw = json_dict["left"]["14"]["quaternion"]["w"]
        wrist_x_end, wrist_y_end, wrist_z_end = get_unit_vector(wrist_x, wrist_y, wrist_z, wrist_qx, wrist_qy, wrist_qz, wrist_qw)
        # 画正视图x
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_x_end[0] + 200, 400 - wrist_x_end[1]), (0, 0, 255), 1)
        # 画正视图y
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_y_end[0] + 200, 400 - wrist_y_end[1]), (0, 255, 0), 1)
        # 画正视图z
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_z_end[0] + 200, 400 - wrist_z_end[1]), (255, 0, 0), 1)
        # 画俯视图x
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_x_end[0] + 200, 200 - wrist_x_end[2]), (0, 0, 255), 1)
        # 画俯视图y
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_y_end[0] + 200, 200 - wrist_y_end[2]), (0, 255, 0), 1)
        # 画俯视图z
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_z_end[0] + 200, 200 - wrist_z_end[2]), (255, 0, 0), 1)


    # 如右手存在，获取右手所有点的位置
    if "right" in json_dict and json_dict["right"]:
        right_points = []
        for i in range(25):
            if str(i) in json_dict["right"]:
                right_position = json_dict["right"][str(i)]["position"]
                right_x = int((right_position["x"]) * 200)
                right_y = int((right_position["y"]) * 200)
                right_z = int((right_position["z"]) * 200)
                cv2.circle(front_view, (right_x + 200, 400 - right_y), 1, (0, 255, 0), -1)
                cv2.circle(top_view, (right_x + 200, 200 - right_z), 1, (0, 255, 0), -1)
                right_points.append([right_x, right_y, right_z])

        #print('right_points:', right_points[0])
        # 画出连接线
        for line in lines:
            for i in range(len(line) - 1):
                index1 = line[i]
                index2 = line[i + 1]
                cv2.line(front_view, (right_points[index1][0] + 200, 400 - right_points[index1][1]), (right_points[index2][0] + 200, 400 - right_points[index2][1]), (0, 255, 0), 2)
                cv2.line(top_view, (right_points[index1][0] + 200, 200 - right_points[index1][2]), (right_points[index2][0] + 200, 200 - right_points[index2][2]), (0, 255, 0), 2)

        # 画手腕的方向
        wrist_x = right_points[0][0]
        wrist_y = right_points[0][1]
        wrist_z = right_points[0][2]
        wrist_qx = json_dict["right"]["0"]["quaternion"]["x"]
        wrist_qy = json_dict["right"]["0"]["quaternion"]["y"]
        wrist_qz = json_dict["right"]["0"]["quaternion"]["z"]
        wrist_qw = json_dict["right"]["0"]["quaternion"]["w"]
        wrist_x_end, wrist_y_end, wrist_z_end = get_unit_vector(wrist_x, wrist_y, wrist_z, wrist_qx, wrist_qy, wrist_qz, wrist_qw)
        # 画正视图x
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_x_end[0] + 200, 400 - wrist_x_end[1]), (0, 0, 255), 1)
        # 画正视图y
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_y_end[0] + 200, 400 - wrist_y_end[1]), (0, 255, 0), 1)
        # 画正视图z
        cv2.line(front_view, (wrist_x + 200, 400 - wrist_y), (wrist_z_end[0] + 200, 400 - wrist_z_end[1]), (255, 0, 0), 1)
        # 画俯视图x
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_x_end[0] + 200, 200 - wrist_x_end[2]), (0, 0, 255), 1)
        # 画俯视图y
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_y_end[0] + 200, 200 - wrist_y_end[2]), (0, 255, 0), 1)
        # 画俯视图z
        cv2.line(top_view, (wrist_x + 200, 200 - wrist_z), (wrist_z_end[0] + 200, 200 - wrist_z_end[2]), (255, 0, 0), 1)



    # 显示拼接图
    combined_view[:, :400] = front_view
    combined_view[:, 400:] = top_view
    #cv2.imshow("Pose", combined_view)
    #cv2.waitKey(1)


    
def get_ryhand_qpos(hand_pose):

    def get_angle(index1, index2):
        wrist_qx = json_dict["left"][str(index1)]["quaternion"]["x"]
        wrist_qy = json_dict["left"][str(index1)]["quaternion"]["y"]
        wrist_qz = json_dict["left"][str(index1)]["quaternion"]["z"]
        wrist_qw = json_dict["left"][str(index1)]["quaternion"]["w"]
        q0 = (wrist_qx, wrist_qy, wrist_qz, wrist_qw)

        wrist_qx = json_dict["left"][str(index2)]["quaternion"]["x"]
        wrist_qy = json_dict["left"][str(index2)]["quaternion"]["y"]
        wrist_qz = json_dict["left"][str(index2)]["quaternion"]["z"]
        wrist_qw = json_dict["left"][str(index2)]["quaternion"]["w"]
        q1 = (wrist_qx, wrist_qy, wrist_qz, wrist_qw)

        axis, angle = get_relative_rotation(q0, q1)

        print(angle)
        if angle<0: angle=0
        if angle>2: angle=2

        angle = float(angle*2000)
        return angle
    
    angle3 = get_angle(5,9)
    angle4 = get_angle(10,14)
    angle5 = get_angle(15,19)
    angle6 = get_angle(20,24)

    hands.set_left_hand([0.0, 0.0, angle3, angle4, angle5, angle6])



# 读取pose.log文件
with open("pose.log", "r") as f:
    for line in f:
        # print(line)
        json_dict = json.loads(line)
        show_pose(json_dict)


# json_dict = json.loads(json_data)
# show_pose(json_dict)
