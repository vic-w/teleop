import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class MotorControlPublisher(Node):
    def __init__(self):
        super().__init__('motor_control_publisher')
        # 创建发布者，发布到 /left_hand_qpos 话题
        self.publisher = self.create_publisher(Float64MultiArray, '/left_hand_qpos', 10)

    def publish_motor_angles(self, angles):
        # 创建 Float64MultiArray 消息并将角度值填充到消息数据中
        msg = Float64MultiArray()
        msg.data = angles

        # 发布消息
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing angles: {angles}')

def main():
    rclpy.init()  # 初始化 ROS 2
    motor_control_publisher = MotorControlPublisher()  # 创建 MotorControlPublisher 节点

    # 角度值 [q1, q2, q3, q4, q5, q6]
    angles = [0, 0, 0, 3000, 0, 0]

    # 在需要时调用函数发布消息
    motor_control_publisher.publish_motor_angles(angles)

    # 关闭 ROS 2 节点
    motor_control_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
