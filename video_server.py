import asyncio
import websockets
import numpy as np
import io
from PIL import Image
import ssl


async def generate_image():
    """生成动态变化的图片"""
    # 创建一个 256x256 的随机图像
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    # 将 NumPy 数组转换为 PIL 图像
    pil_img = Image.fromarray(img)
    # 将 PIL 图像保存到字节流
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'JPEG')  # JPEG 格式
    byte_io.seek(0)
    return byte_io.read()

async def video_stream(websocket, path=None):
    """WebSocket 服务器，持续发送动态图片"""
    try:
        while True:
            # 生成一张新的图片
            image_data = await generate_image()
            # 发送图片数据
            await websocket.send(image_data)
            # 等待一段时间后发送下一张图片
            await asyncio.sleep(0.1)  # 每秒发送 10 张图片
    except websockets.exceptions.ConnectionClosed:
        print("客户端已断开连接")

async def main():
    """启动 WebSocket 服务器"""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
    
    # 启动服务器并通过 wss 协议提供视频流
    server = await websockets.serve(video_stream, '192.168.5.66', 8765, ssl=ssl_context)
    print("WebSocket 服务器已启动，监听 wss://192.168.5.66:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
