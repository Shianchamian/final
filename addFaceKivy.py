import cv2
import dlib
import numpy as np
import os
import time
import sqlite3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock


# 初始化摄像头
current_camera = 0  # 默认使用后置摄像头（根据设备调整索引）


class CameraLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10

        # 摄像头图像显示
        self.image_display = Image(size_hint=(1, 0.6))
        self.add_widget(self.image_display)

        # 输入框
        self.username_input = TextInput(hint_text="Input your name", size_hint=(1, 0.1))
        self.add_widget(self.username_input)

        # 保存图像按钮
        save_button = Button(text="Save Image", size_hint=(1, 0.1))
        save_button.bind(on_press=self.save_image)
        self.add_widget(save_button)

        # 切换摄像头按钮
        switch_button = Button(text="Switch Camera", size_hint=(1, 0.1))
        switch_button.bind(on_press=self.switch_camera)
        self.add_widget(switch_button)

        # 初始化摄像头（OpenCV）
        self.capture = cv2.VideoCapture(0)  # 打开默认摄像头
        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.output_dir = "data"
        self.user_info_db = "user_info.db"  # 用户信息数据库路径

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 创建并连接 SQLite 数据库
        self.conn = sqlite3.connect(self.user_info_db)
        self.cursor = self.conn.cursor()

        # 创建用户信息表（如果表不存在）
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT PRIMARY KEY,
            username TEXT
        )
        """)

        self.conn.commit()

        # 设置每秒更新一次
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 每秒更新30帧

        # 初始化图像计数
        self.image_count = 0

    def update_frame(self, dt):
        """更新摄像头画面"""
        ret, self.frame = self.capture.read()  # 获取摄像头图像
        if ret:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                landmarks = self.sp(self.frame, face)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            buf = cv2.flip(self.frame, 0).tobytes()
            texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_display.texture = texture

    def save_image(self, instance):
        """自动录制3秒并抓拍20张对齐后的人脸图像"""
        self.username = self.username_input.text.strip()  # 获取用户名

        if not self.username:
            print("Username cannot be empty")
            return

        # 生成 user_id
        user_id = self.generate_user_id()

        # 保存用户信息到数据库
        self.save_user_info(user_id, self.username)

        # 开始录制视频并抓拍20张图像
        start_time = time.time()
        end_time = start_time + 3  # 3秒后结束
        frame_interval = 3 / 20  # 每隔一段时间抓拍一张图像，总共20张

        saved_images = 0
        while time.time() < end_time and saved_images < 20:
            ret, self.frame = self.capture.read()  # 获取摄像头图像
            # if not ret:
            #     print("Failed to grab frame.")
            #     break

            # 计算捕获的时间
            elapsed_time = time.time() - start_time
            if elapsed_time >= frame_interval * saved_images:  # 只有在间隔时间达到时才保存图像
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                if faces:
                    for face in faces:
                        if saved_images >= 20:
                            break

                        landmarks = self.sp(self.frame, face)
                        aligned_face = self.align_face(self.frame, landmarks, face)

                        # Save image with proper zero-padding
                        image_path = os.path.join(self.output_dir, f"{user_id}_{saved_images:02d}.jpg")
                        cv2.imwrite(image_path, aligned_face)
                        print(f"Saved to: {image_path}")

                        saved_images += 1

    def save_user_info(self, user_id, username):
        """将用户信息保存到数据库"""
        try:
            # 插入新的用户信息到数据库
            self.cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
            self.conn.commit()
            print(f"User info saved: {user_id}, {username}")
        except sqlite3.IntegrityError:
            # 如果用户 ID 已存在，则打印错误
            print(f"Error: User ID {user_id} already exists.")

    def generate_user_id(self):
        # 获取当前最大用户 ID
        self.cursor.execute("SELECT MAX(user_id) FROM users")
        max_id = self.cursor.fetchone()[0]

        # 如果有现有的 ID，生成下一个 ID
        if max_id is not None:
            return max_id + 1
        else:
            # 如果没有 ID，返回第一个 ID
            return 41

    def align_face(self, frame, landmarks, face_rect, output_size=(160, 160)):
        """对齐人脸"""
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        eye_center = ((left_eye.x + right_eye.x) // 2, (left_eye.y + right_eye.y) // 2)
        dx, dy = right_eye.x - left_eye.x, right_eye.y - left_eye.y
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)
        aligned_face = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        cropped = aligned_face[y:y + h, x:x + w]
        return cv2.resize(cropped, output_size)

    def switch_camera(self, instance):
        """切换前后摄像头"""
        global current_camera
        current_camera = 1 - current_camera  # 切换摄像头索引
        self.capture.release()  # 释放当前摄像头
        self.capture = cv2.VideoCapture(current_camera)  # 打开新摄像头



class AddFaceApp(App):
    def build(self):
        return CameraLayout()


if __name__ == "__main__":
    AddFaceApp().run()
