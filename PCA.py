import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.label import Label
from kivy.clock import Clock

# 导入sklearn模块
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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
        img = Image.open('data\\' + i)
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


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = 10
        self.padding = 20

        # 添加“更新模型”按钮
        self.update_button = Button(text="Update Model", size_hint=(0.5, 0.2))
        self.update_button.bind(on_press=self.update_model)
        self.add_widget(self.update_button)
     

    def update_model(self, instance):
        """当点击按钮时，调用更新模型的函数，并传入进度标签"""
        Clock.schedule_once(lambda dt: update_model(self.update_button), 0)  # 延迟调用模型更新函数



class MyApp(App):
    def build(self):
        return MainScreen()


if __name__ == "__main__":
    MyApp().run()
