# from kivy.app import app
# from kivy.lang import Builder
# from kivy.uix.boxlayout import BoxLayout


# Builder.load_string('''
# <CameraClick>:


# )
# import cv2
# from kivy.app import App
# from kivy.uix.camera import Camera


# class CameraApp(App):
#     def build(self):
#         # Create a camera object
#         self.camera = Camera(resolution=(640, 480), play=True)

#         # Bind the camera to the update_frame method
#         self.camera.bind(on_texture=self.update_frame)

#         # Return the camera object
#         return self.camera

#     def update_frame(self, *args):
#         # Get the camera frame
#         frame1 = self.camera.texture

#         # Convert the frame to a format that Kivy can display
#         buf = frame.pixels
#         buf = buf[::-1]
#         frame.pixels = buf

#         # Display the frame
#         self.camera.texture = frame

# if __name__ == '__main__':
#     CameraApp().run()

# import cv2
# from kivy.app import App
# from kivy.uix.image import Image
# from kivy.clock import Clock
# from kivy.uix.boxlayout import BoxLayout
# from kivy.graphics.texture import Texture


# class DualCameraApp(App):
#     def build(self):
#         # Create two image objects
#         self.img1 = Image()
#         self.img2 = Image()

#         # Schedule the update_frame method to be called every 1/30 seconds
#         Clock.schedule_interval(self.update_frame, 1/30)

#         # Return a layout containing the two image objects
#         return BoxLayout(orientation='horizontal', children=[self.img1, self.img2])

#     def update_frame(self, dt):
#         # Create two VideoCapture objects
#         cap1 = cv2.VideoCapture(0)
#         cap2 = cv2.VideoCapture(1)

#         # Read a frame from each of the video streams
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()

#         # Convert the frames to a format that Kivy can display
#         buf1 = frame1.tobytes()
#         texture1 = Texture.create(size=(frame1.shape[1], frame1.shape[0]), colorfmt='bgr')
#         texture1.blit_buffer(buf1, colorfmt='bgr', bufferfmt='ubyte')

#         buf2 = frame2.tobytes()
#         texture2 = Texture.create(size=(frame2.shape[1], frame2.shape[0]), colorfmt='bgr')
#         texture2.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')

#         # Display the frames in the image objects
#         self.img1.texture = texture1
#         self.img2.texture = texture2

#         # Release the VideoCapture objects
#         cap1.release()
#         cap2.release()

# if __name__ == '__main__':
#     DualCameraApp().run()
# import cv2
# from kivy.app import App
# from kivy.uix.image import Image
# from kivy.clock import Clock
# from kivy.graphics.texture import Texture


# class SingleCameraApp(App):
#     def build(self):
#         # Create an image object
#         self.img = Image()

#         # Schedule the update_frame method to be called every 1/30 seconds
#         Clock.schedule_interval(self.update_frame, 1/30)

#         # Return the image object
#         return self.img

#     def update_frame(self, dt):
#         # Create a VideoCapture object
#         cap = cv2.VideoCapture(0)

#         # Read a frame from the video stream
#         ret, frame = cap.read()

#         # Convert the frame to a format that Kivy can display
#         buf = frame.tobytes()
#         texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
#         texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

#         # Display the frame in the image object
#         self.img.texture = texture

#         # Release the VideoCapture object
#         cap.release()

# if __name__ == '__main__':
#     SingleCameraApp().run()
import os
import datetime
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup


class CameraApp(App):
    def build(self):
        # Create a layout widget to contain the Camera widget
        layout = Widget()

        # Create a Camera widget with a resolution of 640x480 pixels and start playing immediately
        self.camera = Camera(resolution=(640, 480), play=True)

        # Add the Camera widget to the layout widget
        layout.add_widget(self.camera)

        # Schedule the capture_image function to be called every 1 second
        Clock.schedule_interval(self.capture_image, 1)

        # Return the layout widget
        return layout

    def capture_image(self, dt):
        # Get the current date and time
        now = datetime.datetime.now()

        # Generate a filename based on the current date and time
        filename = f"picture_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"

        # Get the full path to the file
        filepath = os.path.join(os.getcwd(), filename)

        # Save the current contents of the Camera widget as a PNG image file
        self.camera.export_to_png(filepath)

        # Create a new Image widget to display the captured image
        image = Image(source=filepath)

        # Create a new Popup widget to display the Image widget
        popup = Popup(title='Captured Image', content=image, size_hint=(0.8, 0.8))

        # Open the Popup widget
        popup.open()

        # Delete the captured image file
        os.remove(filepath)

    def on_stop(self):
        # Stop the Camera widget when the application is closed
        self.camera.stop()

if __name__ == '__main__':
    CameraApp().run()