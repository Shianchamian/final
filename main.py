import sqlite3
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle

# Import necessary classes from other files
from addFaceKivy import AddFaceApp
from face_recognition import Recognition
from PCA import update_model  # Assuming update_model is defined in PCA.py


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set window size and position for mobile
        Window.size = (360, 640)  # Adjust the window size for a typical mobile device
        
        # Create a vertical BoxLayout (this will fill the screen width and height)
        layout = BoxLayout(orientation='vertical', spacing=15, padding=20, size_hint=(1, None), height=500, pos_hint={'center_x': 0.5, 'center_y': 0.5})

        # Set the background color to white
        with self.canvas.before:
            # Color(1, 1, 1, 1)  # White color (R, G, B, A)
            self.rect = Rectangle(size=Window.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)

        # Create buttons with a uniform style
        button_style = {'size_hint': (0.8, None), 'height': 60, 'background_normal': '', 'background_color': (0.2, 0.6, 0.2, 1), 'font_size': 20}

        # Add Face button
        button = Button(text="Add Face", **button_style)
        button.bind(on_press=self.open_add_face)

        # Face Recognition button
        button1 = Button(text="Face Recognition", **button_style)
        button1.bind(on_press=self.open_recognition)

        # Update Model button
        self.update_button = Button(text="Update Model", **button_style)
        self.update_button.bind(on_press=self.update_model)

        # View Database button
        view_db_button = Button(text="View Database", **button_style)
        view_db_button.bind(on_press=self.view_database)

        # Add buttons to the layout
        layout.add_widget(button)
        layout.add_widget(button1)
        layout.add_widget(self.update_button)
        layout.add_widget(view_db_button)

        # Add the layout to the screen
        self.add_widget(layout)

    # Update background when screen is resized
    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def open_add_face(self, instance):
        """Function to switch to AddFaceScreen"""
        self.manager.current = 'add_face'

    def open_recognition(self, instance):
        """Function to switch to RecognitionScreen"""
        self.manager.current = 'recognition'

    def update_model(self, instance):
        """Function to trigger the model update"""
        print("Updating model...")
        # Call the update_model function (you might need to pass the right arguments)
        update_model(self.update_button)  # Make sure this function works properly
        instance.text = "Model Updated"  # Update button text after model is updated

    def view_database(self, instance):
        """Function to switch to ViewDatabaseScreen"""
        self.manager.current = 'view_database'


class AddFaceScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create a vertical BoxLayout to hold the content
        layout = BoxLayout(orientation='vertical', spacing=15, padding=20, size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})

        # Back Button
        back_button = Button(text="Back", size_hint=(None, None), size=(100, 50), pos_hint={'top': 1, 'right': 1})
        back_button.bind(on_press=self.on_back)

        # AddFaceApp content (assuming AddFaceApp has a build method)
        self.add_face_app = AddFaceApp()
        add_face_content = self.add_face_app.build()

        # Add widgets to layout
        layout.add_widget(back_button)
        layout.add_widget(add_face_content)

        self.add_widget(layout)

    def on_back(self, instance):
        self.manager.current = 'main'  # Switch back to the main screen


class RecognitionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create a vertical BoxLayout to hold the content
        layout = BoxLayout(orientation='vertical', spacing=15, padding=20, size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})

        # Back Button
        back_button = Button(text="Back", size_hint=(None, None), size=(100, 50), pos_hint={'top': 1, 'right': 1})
        back_button.bind(on_press=self.on_back)

        # RecognitionApp content (assuming Recognition has a build method)
        self.recognition_app = Recognition()

        # Add widgets to layout
        layout.add_widget(back_button)
        layout.add_widget(self.recognition_app)

        self.add_widget(layout)

    def on_back(self, instance):
        self.manager.current = 'main'  # Switch back to the main screen


class ViewDatabaseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=15, padding=20, size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})

        # Fetch and display user information from the database
        self.db_info = self.get_database_info()

        # Create a ScrollView to display the database info
        scroll_view = ScrollView(size_hint=(1, 1))
        label = Label(text=self.db_info, size_hint_y=None, height=1000)
        scroll_view.add_widget(label)

        # Add ScrollView to layout
        layout.add_widget(scroll_view)

        # Back Button to return to main screen
        back_button = Button(text="Back", size_hint=(None, None), size=(100, 50), pos_hint={'top': 1, 'right': 1})
        back_button.bind(on_press=self.on_back)
        layout.add_widget(back_button)
        

        self.add_widget(layout)

    def get_database_info(self):
        """Fetch all user info from the database and return it as a string"""
        try:
            conn = sqlite3.connect("user_info.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()

            db_info = "User ID | Username\n" + "-"*30 + "\n"
            for user in users:
                db_info += f"{user[0]} | {user[1]}\n"
            conn.close()
            return db_info
        except sqlite3.Error as e:
            return f"Database error: {e}"

    def on_back(self, instance):
        self.manager.current = 'main'  # Switch back to the main screen



class MyApp(App):
    def build(self):
        # Create the ScreenManager
        self.sm = ScreenManager()

        # Add the main screen, AddFaceScreen, RecognitionScreen, and ViewDatabaseScreen
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(AddFaceScreen(name='add_face'))
        self.sm.add_widget(RecognitionScreen(name='recognition'))
        self.sm.add_widget(ViewDatabaseScreen(name='view_database'))

        return self.sm


if __name__ == "__main__":
    MyApp().run()
