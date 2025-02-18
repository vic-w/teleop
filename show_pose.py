import cv2
import numpy as np
import json
from finger_rotation import get_relative_rotation

# json example #头部数据始终存在。左手数据可能不存在。
# {"head":{"position":{"x":0.003198705276239326,"y":1.192264076835687,"z":0.09765001428979647},"quaternion":{"x":0.09823006377827792,"y":-0.17108548383336364,"z":-0.002288945552054866,"w":0.9803445172590474}},"left":{"0":{"position":{"x":-0.1584787368774414,"y":1.2391727683611462,"z":-0.22922492027282715},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"1":{"position":{"x":-0.11672404408454895,"y":1.2586142649419854,"z":-0.21479672193527222},"quaternion":{"x":0.8380455000132265,"y":0.009409801759272455,"z":-0.49795806209868243,"w":0.2227755909639947}},"2":{"position":{"x":-0.08972432464361191,"y":1.2710590255923817,"z":-0.20163491368293762},"quaternion":{"x":0.8231460011391953,"y":0.023486992620617528,"z":-0.4222692384785716,"w":0.37890330193984617}},"3":{"position":{"x":-0.06683357059955597,"y":1.2928089988895008,"z":-0.18959638476371765},"quaternion":{"x":0.8213018649924817,"y":0.024338560763807623,"z":-0.5080916964589244,"w":0.2582899707873113}},"4":{"position":{"x":-0.046632297337055206,"y":1.3031107080646107,"z":-0.1800050437450409},"quaternion":{"x":0.8213018649924817,"y":0.024338560763807623,"z":-0.5080916964589244,"w":0.2582899707873113}},"5":{"position":{"x":-0.12795233726501465,"y":1.263840292162473,"z":-0.21326854825019836},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"6":{"position":{"x":-0.11023347824811935,"y":1.3139066589541981,"z":-0.18551354110240936},"quaternion":{"x":0.7485094717201063,"y":-0.31093829352907365,"z":-0.03052592943240229,"w":0.5849094938490937}},"7":{"position":{"x":-0.09470455348491669,"y":1.3463965994782994,"z":-0.1736082136631012},"quaternion":{"x":0.6478676847068099,"y":-0.3208723219595374,"z":-0.05715956405272852,"w":0.6885064998605565}},"8":{"position":{"x":-0.08216606825590134,"y":1.3671869379945347,"z":-0.1725052446126938},"quaternion":{"x":0.6239693677154499,"y":-0.3121267916800058,"z":-0.03821285141358687,"w":0.7153872182637674}},"9":{"position":{"x":-0.07169289886951447,"y":1.3869737727113316,"z":-0.17227570712566376},"quaternion":{"x":0.6239693677154499,"y":-0.3121267916800058,"z":-0.03821285141358687,"w":0.7153872182637674}},"10":{"position":{"x":-0.1423814594745636,"y":1.2687178818054745,"z":-0.21943213045597076},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"11":{"position":{"x":-0.1303846389055252,"y":1.322102804250295,"z":-0.19061076641082764},"quaternion":{"x":0.7122921518455835,"y":-0.30122034554434773,"z":0.04741060423254484,"w":0.6321854383451401}},"12":{"position":{"x":-0.11693500727415085,"y":1.3619890255399296,"z":-0.18218903243541718},"quaternion":{"x":0.5765147757812714,"y":-0.31209621266310017,"z":-0.0009353394300958266,"w":0.7551329634490669}},"13":{"position":{"x":-0.10391984134912491,"y":1.3859600765176365,"z":-0.18605847656726837},"quaternion":{"x":0.5254823559311322,"y":-0.34018435514362405,"z":0.020179278856521907,"w":0.7795740470445475}},"14":{"position":{"x":-0.09188731014728546,"y":1.4073656541772435,"z":-0.1907120943069458},"quaternion":{"x":0.5254823559311322,"y":-0.34018435514362405,"z":0.020179278856521907,"w":0.7795740470445475}},"15":{"position":{"x":-0.15806567668914795,"y":1.277347785002763,"z":-0.22564782202243805},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"16":{"position":{"x":-0.14548444747924805,"y":1.325401474303777,"z":-0.20452642440795898},"quaternion":{"x":0.7031476870057352,"y":-0.3082189240257777,"z":0.11711951969231534,"w":0.6299741607700414}},"17":{"position":{"x":-0.1367635428905487,"y":1.3627646905847142,"z":-0.19755271077156067},"quaternion":{"x":0.5635478907127625,"y":-0.3527339523171017,"z":0.07972596965624354,"w":0.7427222250063107}},"18":{"position":{"x":-0.12522782385349274,"y":1.3865043861337254,"z":-0.20063483715057373},"quaternion":{"x":0.4847371206363514,"y":-0.3860662902399787,"z":0.034037342798131524,"w":0.7841072648003987}},"19":{"position":{"x":-0.11217083036899567,"y":1.4065672201581547,"z":-0.20526210963726044},"quaternion":{"x":0.4847371206363514,"y":-0.3860662902399787,"z":0.034037342798131524,"w":0.7841072648003987}},"20":{"position":{"x":-0.16285032033920288,"y":1.2810313147969792,"z":-0.23195011913776398},"quaternion":{"x":0.6767120887505773,"y":-0.3888533711392298,"z":0.2586043324381901,"w":0.5691903055540259}},"21":{"position":{"x":-0.15861821174621582,"y":1.3253791896172116,"z":-0.22198377549648285},"quaternion":{"x":0.6415165695796935,"y":-0.3172653795417424,"z":0.21510603067819334,"w":0.6644761586880363}},"22":{"position":{"x":-0.1541440188884735,"y":1.3557628137536595,"z":-0.22123414278030396},"quaternion":{"x":0.5091149143532465,"y":-0.3543861549828658,"z":0.22200270357574306,"w":0.752281368069559}},"23":{"position":{"x":-0.14790543913841248,"y":1.3745172066159794,"z":-0.22591440379619598},"quaternion":{"x":0.48091358433447695,"y":-0.4012128213020061,"z":0.17782075385283033,"w":0.7590323945159795}},"24":{"position":{"x":-0.1389487087726593,"y":1.3941979748673985,"z":-0.22973212599754333},"quaternion":{"x":0.48091358433447695,"y":-0.4012128213020061,"z":0.17782075385283033,"w":0.7590323945159795}}}}

# 连接手指关节编号，生成连接线
lines = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8, 9],
    [0, 10, 11, 12, 13, 14],
    [0, 15, 16, 17, 18, 19],
    [0, 20, 21, 22, 23, 24]
]

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
    cv2.imshow("Pose", combined_view)
    cv2.waitKey(1)


    
def get_ryhand_qpos(hand_pose):
    wrist_qx = json_dict["left"]["10"]["quaternion"]["x"]
    wrist_qy = json_dict["left"]["10"]["quaternion"]["y"]
    wrist_qz = json_dict["left"]["10"]["quaternion"]["z"]
    wrist_qw = json_dict["left"]["10"]["quaternion"]["w"]
    q0 = (wrist_qx, wrist_qy, wrist_qz, wrist_qw)

    wrist_qx = json_dict["left"]["14"]["quaternion"]["x"]
    wrist_qy = json_dict["left"]["14"]["quaternion"]["y"]
    wrist_qz = json_dict["left"]["14"]["quaternion"]["z"]
    wrist_qw = json_dict["left"]["14"]["quaternion"]["w"]
    q1 = (wrist_qx, wrist_qy, wrist_qz, wrist_qw)

    axis, angle = get_relative_rotation(q0, q1)
    print(angle)
    if angle<0: angle=0
    if angle>2: angle=2

    angle = int(angle*2000)
    



# 读取pose.log文件
with open("pose.log", "r") as f:
    for line in f:
        # print(line)
        json_dict = json.loads(line)
        show_pose(json_dict)

# 示例json数据
# json_data = '{"head":{"position":{"x":0.003198705276239326,"y":1.192264076835687,"z":0.09765001428979647},"quaternion":{"x":0.09823006377827792,"y":-0.17108548383336364,"z":-0.002288945552054866,"w":0.9803445172590474}},"left":{"0":{"position":{"x":-0.1584787368774414,"y":1.2391727683611462,"z":-0.22922492027282715},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"1":{"position":{"x":-0.11672404408454895,"y":1.2586142649419854,"z":-0.21479672193527222},"quaternion":{"x":0.8380455000132265,"y":0.009409801759272455,"z":-0.49795806209868243,"w":0.2227755909639947}},"2":{"position":{"x":-0.08972432464361191,"y":1.2710590255923817,"z":-0.20163491368293762},"quaternion":{"x":0.8231460011391953,"y":0.023486992620617528,"z":-0.4222692384785716,"w":0.37890330193984617}},"3":{"position":{"x":-0.06683357059955597,"y":1.2928089988895008,"z":-0.18959638476371765},"quaternion":{"x":0.8213018649924817,"y":0.024338560763807623,"z":-0.5080916964589244,"w":0.2582899707873113}},"4":{"position":{"x":-0.046632297337055206,"y":1.3031107080646107,"z":-0.1800050437450409},"quaternion":{"x":0.8213018649924817,"y":0.024338560763807623,"z":-0.5080916964589244,"w":0.2582899707873113}},"5":{"position":{"x":-0.12795233726501465,"y":1.263840292162473,"z":-0.21326854825019836},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"6":{"position":{"x":-0.11023347824811935,"y":1.3139066589541981,"z":-0.18551354110240936},"quaternion":{"x":0.7485094717201063,"y":-0.31093829352907365,"z":-0.03052592943240229,"w":0.5849094938490937}},"7":{"position":{"x":-0.09470455348491669,"y":1.3463965994782994,"z":-0.1736082136631012},"quaternion":{"x":0.6478676847068099,"y":-0.3208723219595374,"z":-0.05715956405272852,"w":0.6885064998605565}},"8":{"position":{"x":-0.08216606825590134,"y":1.3671869379945347,"z":-0.1725052446126938},"quaternion":{"x":0.6239693677154499,"y":-0.3121267916800058,"z":-0.03821285141358687,"w":0.7153872182637674}},"9":{"position":{"x":-0.07169289886951447,"y":1.3869737727113316,"z":-0.17227570712566376},"quaternion":{"x":0.6239693677154499,"y":-0.3121267916800058,"z":-0.03821285141358687,"w":0.7153872182637674}},"10":{"position":{"x":-0.1423814594745636,"y":1.2687178818054745,"z":-0.21943213045597076},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"11":{"position":{"x":-0.1303846389055252,"y":1.322102804250295,"z":-0.19061076641082764},"quaternion":{"x":0.7122921518455835,"y":-0.30122034554434773,"z":0.04741060423254484,"w":0.6321854383451401}},"12":{"position":{"x":-0.11693500727415085,"y":1.3619890255399296,"z":-0.18218903243541718},"quaternion":{"x":0.5765147757812714,"y":-0.31209621266310017,"z":-0.0009353394300958266,"w":0.7551329634490669}},"13":{"position":{"x":-0.10391984134912491,"y":1.3859600765176365,"z":-0.18605847656726837},"quaternion":{"x":0.5254823559311322,"y":-0.34018435514362405,"z":0.020179278856521907,"w":0.7795740470445475}},"14":{"position":{"x":-0.09188731014728546,"y":1.4073656541772435,"z":-0.1907120943069458},"quaternion":{"x":0.5254823559311322,"y":-0.34018435514362405,"z":0.020179278856521907,"w":0.7795740470445475}},"15":{"position":{"x":-0.15806567668914795,"y":1.277347785002763,"z":-0.22564782202243805},"quaternion":{"x":0.7823083318416001,"y":-0.3114688256300533,"z":0.044645737689814946,"w":0.5375756716016644}},"16":{"position":{"x":-0.14548444747924805,"y":1.325401474303777,"z":-0.20452642440795898},"quaternion":{"x":0.7031476870057352,"y":-0.3082189240257777,"z":0.11711951969231534,"w":0.6299741607700414}},"17":{"position":{"x":-0.1367635428905487,"y":1.3627646905847142,"z":-0.19755271077156067},"quaternion":{"x":0.5635478907127625,"y":-0.3527339523171017,"z":0.07972596965624354,"w":0.7427222250063107}},"18":{"position":{"x":-0.12522782385349274,"y":1.3865043861337254,"z":-0.20063483715057373},"quaternion":{"x":0.4847371206363514,"y":-0.3860662902399787,"z":0.034037342798131524,"w":0.7841072648003987}},"19":{"position":{"x":-0.11217083036899567,"y":1.4065672201581547,"z":-0.20526210963726044},"quaternion":{"x":0.4847371206363514,"y":-0.3860662902399787,"z":0.034037342798131524,"w":0.7841072648003987}},"20":{"position":{"x":-0.16285032033920288,"y":1.2810313147969792,"z":-0.23195011913776398},"quaternion":{"x":0.6767120887505773,"y":-0.3888533711392298,"z":0.2586043324381901,"w":0.5691903055540259}},"21":{"position":{"x":-0.15861821174621582,"y":1.3253791896172116,"z":-0.22198377549648285},"quaternion":{"x":0.6415165695796935,"y":-0.3172653795417424,"z":0.21510603067819334,"w":0.6644761586880363}},"22":{"position":{"x":-0.1541440188884735,"y":1.3557628137536595,"z":-0.22123414278030396},"quaternion":{"x":0.5091149143532465,"y":-0.3543861549828658,"z":0.22200270357574306,"w":0.752281368069559}},"23":{"position":{"x":-0.14790543913841248,"y":1.3745172066159794,"z":-0.22591440379619598},"quaternion":{"x":0.48091358433447695,"y":-0.4012128213020061,"z":0.17782075385283033,"w":0.7590323945159795}},"24":{"position":{"x":-0.1389487087726593,"y":1.3941979748673985,"z":-0.22973212599754333},"quaternion":{"x":0.48091358433447695,"y":-0.4012128213020061,"z":0.17782075385283033,"w":0.7590323945159795}}}}'


# json_dict = json.loads(json_data)
# show_pose(json_dict)