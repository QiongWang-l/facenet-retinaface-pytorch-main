import sys
import cv2
import numpy as np
import torch
import lowlight_model
from retinaface import Retinaface
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from pyqt5_tools.examples.exampleqmlitem import QtCore
from QT.untitled import Ui_LLF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


def enhance_and_reg(image, lle_net, retinaface):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 进行增强
    frame = frame / 255.0
    frame = torch.from_numpy(frame).to('cuda', dtype=torch.float32)  # 转化为tensor
    frame = frame.permute(2, 0, 1)  # 维度转置 C*H*W*
    frame = frame.unsqueeze(0)  # 在第0维插入1个维度
    _, frame, _ = lle_net(frame)
    frame = frame.detach().cpu().squeeze(0)
    frame = frame.permute(1, 2, 0).numpy()  # 维度转置 H*W*C
    frame = frame * 255.0
    # 进行检测
    frame, facename = np.array(retinaface.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame, facename


class LLF(QWidget, Ui_LLF):
    def __init__(self):
        super(LLF, self).__init__()
        self.facename = None
        self.enhancedImage = None
        self.image = None
        self.timer = None
        self.capture = None
        self.setupUi(self)
        self.setWindowTitle('低质人脸增强和识别系统')

        self.startButton.clicked.connect(self.start_cam)
        self.stopButton.clicked.connect(self.stop_cam)
        self.enhanceButton.toggled.connect(self.enhance_cam)
        self.enhanceButton.setCheckable(True)
        self.enhance_Enable = False

    def play(self):
        try:
            self.video.captureNextFrame()
            self.videoFrame.setPixmap(self.video.convertFrame())
            self.videoFrame.setScaledContents(True)  # 设置图像自动填充控件
        except TypeError:
            print('No Frame')

    def enhance_cam(self, status):
        if status:
            self.enhance_Enable = True
            self.enhanceButton.setText("Stop Enhance")
        else:
            self.enhance_Enable = False
            self.enhanceButton.setText("Enhance")
            name_str = ''
            for name in self.facename:
                name_str += name.replace('.jpg', '') + ','
            self.label_2.setText(name_str)

    def start_cam(self):
        camera_number = 0
        self.capture = cv2.VideoCapture(0)
        # 这里若写self.capture = cv2.VideoCapture(0)会报错
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)  # 翻转图像 镜像
        self.displayImage(self.image, 1)

        if self.enhance_Enable:
            self.enhancedImage, self.facename = enhance_and_reg(self.image, LLE_net, retinaface)
            self.displayImage(self.enhancedImage, 2)

    def stop_cam(self):
        self.timer.stop()

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if (len(img.shape)) == 3:  # [0]=rows,[1]=cols,[2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR->RGB
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.original_video.setPixmap(QPixmap.fromImage(outImage))
            self.original_video.setScaledContents(True)
        if window == 2:
            self.enhanced_video.setPixmap(QPixmap.fromImage(outImage))
            self.enhanced_video.setScaledContents(True)


if __name__ == '__main__':
    retinaface = Retinaface()
    LLE_net = lowlight_model.enhance_net_nopool().cuda()
    LLE_net.load_state_dict(torch.load('model_data/LLE.pth'))
    print('LLE_net load weights from LLE.pth')
    
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 解决了Qtdesigner设计的界面与实际运行界面不一致的问题
    # application 对象
    app = QApplication(sys.argv)
    llf_window = LLF()
    llf_window.show()

    '''
    capture = cv2.VideoCapture(0)  # 0为调用电脑摄像头
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, frame_d = frame.shape  # 获取图像的高，宽以及深度。
    qt_frame = QImage(frame.data, frame_w, frame_h, QImage.Format_RGB888)
    llf_window.original_video.setPixmap(QPixmap.fromImage(qt_frame))
    llf_window.original_video.setScaledContents(True)
    # llf_window.original_video.show()
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        capture.release()
    '''

    sys.exit(app.exec_())
