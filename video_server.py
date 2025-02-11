import asyncio
import websockets
import ssl
import struct
from PIL import Image, ImageDraw
import io
import random


class ImageStream:
    def __init__(self, left_image_path, right_image_path):
        self.left_image = Image.open(left_image_path)
        self.right_image = Image.open(right_image_path)
        
        # 方块的初始颜色
        self.left_square_color = (255, 0, 0)  # 红色
        self.right_square_color = (0, 255, 0)  # 绿色

        # 方块的大小（例如 50x50）
        self.square_size = 50

    def change_square_color(self):
        """每次改变方块的颜色"""
        # 随机生成 RGB 颜色
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.left_square_color = (r, g, b)
        self.right_square_color = (r, g, b)

    def add_square_to_image(self, image, square_color):
        """在图像中心添加一个小方块"""
        draw = ImageDraw.Draw(image)
        # 计算方块的左上角和右下角
        width, height = image.size
        left = (width - self.square_size) // 2
        top = (height - self.square_size) // 2
        right = left + self.square_size
        bottom = top + self.square_size

        # 绘制方块
        draw.rectangle([left, top, right, bottom], fill=square_color)

    def get_image_data(self):
        """返回加了方块的图像字节流"""
        # 为左图和右图添加方块
        left_image_with_square = self.left_image.copy()
        right_image_with_square = self.right_image.copy()

        self.add_square_to_image(left_image_with_square, self.left_square_color)
        self.add_square_to_image(right_image_with_square, self.right_square_color)

        # 将处理后的图片转换为字节流
        left_image_data = io.BytesIO()
        left_image_with_square.save(left_image_data, format='PNG')
        left_image_data = left_image_data.getvalue()

        right_image_data = io.BytesIO()
        right_image_with_square.save(right_image_data, format='PNG')
        right_image_data = right_image_data.getvalue()

        return left_image_data, right_image_data


async def generate_images(image_stream):
    """获取加了方块并改变颜色后的图片数据"""
    image_stream.change_square_color()
    return image_stream.get_image_data()


async def video_stream(websocket, path=None):
    """WebSocket 服务器，持续发送包含小方块并改变颜色的 left.png 和 right.png 图片"""
    image_stream = ImageStream('left.png', 'right.png')  # 初始化图像流
    try:
        while True:
            # 获取加了方块并改变颜色后的图片数据
            left_image_data, right_image_data = await generate_images(image_stream)

            # 计算两张图片的长度
            left_length = len(left_image_data)
            right_length = len(right_image_data)

            # 创建包头，包含两张图片的长度（每个长度使用 4 个字节）
            header = struct.pack('!II', left_length, right_length)

            # 发送包头 + 左图数据 + 右图数据
            await websocket.send(header + left_image_data + right_image_data)

            # 等待一段时间后发送下一次图片
            await asyncio.sleep(0.1)  # 每秒发送 10 次，左图和右图一次
    except websockets.exceptions.ConnectionClosed:
        print("客户端已断开连接")


async def main():
    """启动 WebSocket 服务器"""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
    
    # 启动服务器并通过 wss 协议提供视频流
    server = await websockets.serve(video_stream, '0.0.0.0', 8765, ssl=ssl_context)
    print("WebSocket 服务器已启动，监听 wss://0.0.0.0:8765")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
