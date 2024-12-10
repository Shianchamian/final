from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import dlib
import os
import sqlite3
import time
import numpy as np
import pandas as pd
import joblib
from PIL import Image as PILImage  # 避免与 Kivy 的 Image 冲突
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
KV ="""
<MainScreen>:
    name: "main_screen"

    MDTopAppBar:
        elevation: 10
        left_action_items: [["menu", lambda x: app.navigation_draw()]]
    
    MDBottomNavigation:
        panel_color: 0, 0, 0, 1

        MDBottomNavigationItem:
            name: "add_face"
            text: "Add Face"
            icon: "account-plus"
            on_tab_press: app.change_screen("add_face_screen")

            AddFaceScreen:

        MDBottomNavigationItem:
            name: "update_model"
            text: "Update Model"
            icon: "update"
            on_tab_press: app.change_screen("update_model_screen")

            UpdateModelScreen:




   """

# 加载PCA和SVM模型
pca = joblib.load('model/pca_model.pkl')  # 加载PCA模型
svc = joblib.load('model/svc_model.pkl')  # 加载SVM模型

# 加载dlib的人脸检测器和形状预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 初始化摄像头
current_camera = 0  # 当前摄像头索引

# 定义数据库路径和数据保存路径
USER_INFO_DB = "user_info.db"
OUTPUT_DIR = "data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 创建SQLite数据库连接
conn = sqlite3.connect(USER_INFO_DB)
cursor = conn.cursor()

# 创建用户信息表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT
)
""")
conn.commit()



class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical")
        
        # 添加标签，表示 MainScreen
        label = MDLabel(text="Welcome to Face Recognition App!", halign="center", font_style="H4")
        
        # 添加一个按钮，点击时切换到 'add_face_screen'
        button = MDRaisedButton(text="Go to Add Face", size_hint=(None, None), size=("200dp", "50dp"), pos_hint={"center_x": 0.5})
        button.bind(on_press=lambda x: self.change_screen("add_face_screen"))

        layout.add_widget(label)
        layout.add_widget(button)
        
        self.add_widget(layout)
    
    def change_screen(self, screen_name):
        self.manager.current = screen_name
        

# 数据和模型更新的功能
def update_model(update_button):
    # 关闭警告
    warnings.filterwarnings("ignore")

    # 获取图片文件名
    names = os.listdir('data')  # 返回指定文件夹下的文件或文件夹的名称列表

    # 提取特征变量
    X = []  # 存储所有图片的特征
    for idx, i in enumerate(names):
        # 读取图片并转为灰度
        img = PILImage.open('data\\' + i)
        img = img.convert('L')  # 转为灰度图
        img = img.resize((32, 32))  # 调整为32x32尺寸
        arr = np.array(img).reshape(1, -1)  # 转为一维数组
        X.append(arr.flatten().tolist())  # 将特征值添加到X

    X = pd.DataFrame(X)  # 转换为DataFrame

    # 目标变量
    y = []
    for i in names:
        y.append(int(i.split('_')[0]))  # 从文件名中提取标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PCA降维
    pca = PCA(n_components=100)
    pca.fit(X_train)  # 使用训练集拟合PCA模型

    X_train_pca = pca.transform(X_train)  # 对训练集进行降维
    X_test_pca = pca.transform(X_test)  # 对测试集进行降维

    # SVM模型训练
    svc = SVC()
    svc.fit(X_train_pca, y_train)  # 使用训练集训练SVM

    # 模型预测
    y_pred = svc.predict(X_test_pca)

    # 分类报告
    # print(classification_report(y_test, y_pred))

    # 确保保存目录存在
    save_dir = 'model'
    os.makedirs(save_dir, exist_ok=True)

    # 保存PCA和SVM模型
    joblib.dump(pca, os.path.join(save_dir, 'pca_model.pkl'))  # 保存PCA模型
    joblib.dump(svc, os.path.join(save_dir, 'svc_model.pkl'))  # 保存SVM模型

    # 更新按钮文本为 "Updated"
    update_button.text = "Updated"

class AddFaceScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.capture = None
        self.layout = BoxLayout(orientation="vertical")

        # 添加UI组件
        self.image_display = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.image_display)

        self.username_input = TextInput(hint_text="Input your name", size_hint=(1, 0.1))
        self.layout.add_widget(self.username_input)

        save_button = MDRaisedButton(text="Save Face", size_hint=(1, 0.1))
        save_button.bind(on_press=self.save_face)
        self.layout.add_widget(save_button)

        self.switch_button = MDRaisedButton(text="Switch Camera", size_hint=(1, 0.1))
        self.switch_button.bind(on_press=self.switch_camera)
        self.layout.add_widget(self.switch_button)

        self.add_widget(self.layout)

    def switch_camera(self, instance):
        """切换摄像头"""
        self.capture.release()
        self.capture = cv2.VideoCapture(1 if self.capture.get(cv2.CAP_PROP_POS_FRAMES) == 0 else 0)

    def save_face(self, instance):
        """保存人脸图像到数据库"""
        username = self.username_input.text.strip()
        if not username:
            print("Username cannot be empty!")
            return

        user_id = self.generate_user_id()
        self.save_user_info(user_id, username)

        # 录制并保存20张人脸图像
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            return

        saved_images = 0
        start_time = time.time()
        while time.time() - start_time < 3 and saved_images < 20:
            ret, frame = self.capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            for face in faces:
                landmarks = self.sp(frame, face)
                aligned_face = self.align_face(frame, landmarks, face)
                image_path = os.path.join(OUTPUT_DIR, f"{user_id}_{saved_images:02d}.jpg")
                cv2.imwrite(image_path, aligned_face)
                saved_images += 1
                print(f"Saved image {saved_images}: {image_path}")

        self.capture.release()

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

    def generate_user_id(self):
        """生成新的用户ID"""
        cursor.execute("SELECT MAX(user_id) FROM users")
        max_id = cursor.fetchone()[0]
        return max_id + 1 if max_id else 1

    def save_user_info(self, user_id, username):
        """将用户信息保存到数据库"""
        try:
            cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
            conn.commit()
        except sqlite3.IntegrityError:
            print("Error: User already exists!")


class UpdateModelScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="vertical", spacing=10)
        
        self.label = MDLabel(text="Update Recognition Model", halign="center", font_style="H5")
        self.update_button = MDRaisedButton(text="Update Model", size_hint=(None, None), size=("200dp", "50dp"), pos_hint={"center_x": 0.5})
        
        layout.add_widget(self.label)
        layout.add_widget(self.update_button)
        
        self.add_widget(layout)

# Kivy应用
class MainApp(MDApp):
    def build(self):
        print("Building the UI...")
        screen_manager = ScreenManager()

        # 添加屏幕
        screen_manager.add_widget(MainScreen(name="main_screen"))
        screen_manager.add_widget(AddFaceScreen(name="add_face_screen"))
        screen_manager.add_widget(UpdateModelScreen(name="update_model_screen"))
        
        return screen_manager
    
    def change_screen(self, screen_name):
        print(f"Changing screen to {screen_name}")
        self.root.current = screen_name


if __name__ == '__main__':
    MainApp().run()
