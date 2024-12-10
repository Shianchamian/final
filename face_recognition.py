import cv2
import dlib
import numpy as np
from PIL import Image as PILImage  # 避免与 Kivy 的 Image 冲突
import joblib
import sqlite3
from kivy.app import App
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image  # 使用 Kivy 的 Image

# 加载PCA和SVM模型
pca = joblib.load('model/pca_model.pkl')  # 加载PCA模型
svc = joblib.load('model/svc_model.pkl')  # 加载SVM模型

# 加载dlib的人脸检测器和形状预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

# 初始化摄像头
current_camera = 0  # 当前摄像头索引

# Database connection
conn = sqlite3.connect('user_info.db')  # Connect to the database
cursor = conn.cursor()



class Recognition(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"


        # 添加图像组件
        self.image = Image(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.image)

        # 打开摄像头
        self.capture = cv2.VideoCapture(current_camera)
        Clock.schedule_interval(self.update_frame, 1.0 / 30)  # 每秒更新30帧

    def update_frame(self, dt):
        """更新摄像头画面"""
        ret, frame = self.capture.read()
        if ret:
            # 转为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = detector(gray)

            for face in faces:
                # 获取人脸特征点
                landmarks = predictor(gray, face)

                # 绘制人脸框
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 获取人脸区域并调整大小
                face_region = gray[y:y + h, x:x + w]
                img = PILImage.fromarray(face_region)
                img = img.resize((32, 32))
                arr = np.array(img).reshape(1, -1)

                # 使用PCA降维
                arr_pca = pca.transform(arr)

                # 使用SVM进行预测
                predicted_label = svc.predict(arr_pca)

                 # Fetch username based on predicted label from the database
                username = self.get_username_from_db(predicted_label[0])

                # 显示预测结果
                cv2.putText(frame, f"{username}{predicted_label}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # 将 OpenCV 图像转换为 Kivy 的纹理
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
        
    def get_username_from_db(self, predicted_label):

        predicted_label = int(predicted_label)

        """Fetch username from the database based on predicted label"""
        cursor.execute("SELECT username FROM users WHERE user_id = ?", (predicted_label,))
        result = cursor.fetchone()

        if result:
            return result[0]  # Return the username
        else:
            return "Unknown"  # If not found, return 'Unknown'

    def switch_camera(self):
        """切换前后摄像头"""
        global current_camera
        current_camera = 1 - current_camera
        self.capture.release()
        self.capture = cv2.VideoCapture(current_camera)

    def stop(self):
        """停止摄像头"""
        self.capture.release()


class FaceRecognitionApp(MDApp):
    def build(self):
        return Recognition()


if __name__ == "__main__":
    FaceRecognitionApp().run()
