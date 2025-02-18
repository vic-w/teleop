import asyncio
import websockets
import ssl

logfp = open('pose.log', 'w')

# 处理客户端连接的异步函数
async def handle_connection(websocket, path=None):
    print(f"New connection from {path}")
    try:
        async for message in websocket:
            print(f"Received: {message}")
            logfp.write(f"{message}\n")
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
