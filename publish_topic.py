import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class MotorControlPublisher(Node):
    def __init__(self):
        super().__init__('motor_control_publisher')
        # 创建发布者，发布到 /left_hand_qpos 话题
        self.left_publisher = self.create_publisher(Float64MultiArray, '/set_left_hand_qpos', 10)
        self.right_publisher = self.create_publisher(Float64MultiArray, '/set_right_hand_qpos', 10)

    def publish_left_hand_angles(self, angles):
        # 创建 Float64MultiArray 消息并将角度值填充到消息数据中
        msg = Float64MultiArray()
        msg.data = angles

        # 发布消息
        self.left_publisher.publish(msg)
        #self.get_logger().info(f'Publishing angles: {angles}')

    def publish_right_hand_angles(self, angles):
        # 创建 Float64MultiArray 消息并将角度值填充到消息数据中
        msg = Float64MultiArray()
        msg.data = angles

        # 发布消息
        self.right_publisher.publish(msg)
        #self.get_logger().info(f'Publishing angles: {angles}')

class hand_controller:
    def __init__(self):
        rclpy.init()  # 初始化 ROS 2
        self.motor_control_publisher = MotorControlPublisher()  # 创建 MotorControlPublisher 节点

    def set_left_hand(self, angles):
        # 在需要时调用函数发布消息
        self.motor_control_publisher.publish_left_hand_angles(angles)

    def set_right_hand(self, angles):
        # 在需要时调用函数发布消息
        self.motor_control_publisher.publish_right_hand_angles(angles)

    def __del__(self):
        # 关闭 ROS 2 节点
        #self.motor_control_publisher.destroy_node()
        #rclpy.shutdown()
        pass

if __name__ == '__main__':
    hands = hand_controller()

    angles = [0.0, 0.0, 0.0, 3000.0, 0.0, 0.0]

    hands.set_left_hand(angles)
