
import time

import cv2
import numpy as np
import torch
from PIL import Image

import lowlight_model
from retinaface import Retinaface
from lowlight_test import lowlight

if __name__ == "__main__":
    retinaface = Retinaface()
    mode = "video"
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    LLE_net = lowlight_model.enhance_net_nopool().cuda()
    LLE_net.load_state_dict(torch.load('model_data/LLE.pth'))
    print('LLE_net load weights from LLE.pth')

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            # img = 'img/1.jpg'
            # img = lowlight(img)
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("after", r_image)
                cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)  # 0为调用电脑摄像头
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.1
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 增强
            # frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


            frame = frame / 255.0
            frame = torch.from_numpy(frame).to('cuda', dtype=torch.float32)  # 转化为tensor
            frame = frame.permute(2, 0, 1)  # 维度转置 C*H*W*
            frame = frame.unsqueeze(0)  # 在第0维插入1个维度
            _, frame, _ = LLE_net(frame)
            frame = frame.detach().cpu().squeeze(0)
            frame = frame.permute(1, 2, 0).numpy()  # 维度转置 H*W*C
            frame = frame * 255.0

            # 进行检测
            frame, name = np.array(retinaface.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                # image_path = lowlight(image_path)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
