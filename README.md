# teleop

本程序使用WebXR实现机器人的遥操作

运行
`sudo python https_server.py`

必须使用https协议建立服务器

从oculus里面打开网页浏览器。
输入https://加ip地址
即可看到WebRX网页，此时点击"Enter VR"即可进入沉浸式体验模式


https://immersive-web.github.io/webxr-hand-input/
WebXR关于hand动作的数据解析


## 基于FastAPI将KingFisher图像去畸变并串流到oculus
### 前提
1. 安装KingFisher SDK：https://gitee.com/open3dv/kingfisher-r-6000/releases
2. 安装fastapi等python包：\
`pip install "fastapi[all]"`\
`pip install uvicorn`\
`pip install websockets`\

### 使用步骤
1. 修改程序`kingfisher_server_websockets.py`中相机ip：\
c=kingfisher.connect(`"192.168.9.125"`)

2. 运行`https_server.py`，然后再运行`kingfisher_server_websockets.py`\

3. 在电脑浏览器中输入`https://加ip地址/video.html`即可预览图像；\

4. oculus浏览器中输入`https://加ip地址`即可看到WebRX网页，此时点击"Enter VR"即可进入沉浸式体验模式；

5. 另外支持FastAPI接口：浏览器输入`https://ip地址:8012/kingfisher/`可以调用FastAPI中定义的`kingfisher`功能，显示相机视频流；

### 遗留问题
1. 程序中的标定参数为读取的标定参数，可以根据需要自动从相机获取；

