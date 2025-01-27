import http.server
import ssl

# 服务器地址和端口
server_address = ('0.0.0.0', 443)  # 0.0.0.0 表示监听所有网络接口
# 证书和私钥文件路径
certfile = 'cert.pem'
keyfile = 'key.pem'

# 创建 HTTPS 服务器
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(
    httpd.socket,
    server_side=True,
    certfile=certfile,
    keyfile=keyfile,
    ssl_version=ssl.PROTOCOL_TLS
)

print(f"HTTPS 服务器已启动，访问 https://localhost:{server_address[1]}")
httpd.serve_forever()
