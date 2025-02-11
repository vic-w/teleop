import asyncio
import websockets
import ssl
import struct


async def generate_images():
    """从文件读取 left.png 和 right.png 图片并将其转换为字节流"""
    # 读取 left.png 图片
    with open('left.png', 'rb') as f:
        left_image_data = f.read()
    
    # 读取 right.png 图片
    with open('right.png', 'rb') as f:
        right_image_data = f.read()

    return left_image_data, right_image_data

async def video_stream(websocket, path=None):
    """WebSocket 服务器，持续发送 left.png 和 right.png 图片"""
    try:
        while True:
            # 获取 left.png 和 right.png 图片数据
            left_image_data, right_image_data = await generate_images()

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
