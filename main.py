from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import  Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from android.permissions import request_permissions, Permission 
from FaceDetector import FaceDetection
from FaceRecognizer import FaceRecognizer

import numpy as np
import cv2
import time
import kivy


class MyCamera(Camera):
    DETECTION_TIME = 10

    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        self.faceDetector = FaceDetection()
        self.recognizer = FaceRecognizer()
        self.detectionFrame = 0

    def _camera_loaded(self, *largs):
        if kivy.platform == 'android':
            self.texture = Texture.create(size=self.resolution, colorfmt='rgb')
            self.texture_size = list(self.texture.size)
        else:
            self.texture = self._camera.texture
            self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        if kivy.platform == 'android':
            buf = self._camera.grab_frame()
            if buf is None:
                return
            frame = self._camera.decode_frame(buf)
        else:
            ret, frame = self._camera._device.read()
        if frame is None:
            print("No")

        buf = self.process_frame(frame)
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        super(MyCamera, self).on_tex(*l)

    def process_frame(self, frame):
        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.detectionFrame == self.DETECTION_TIME:
            self.detectionFrame = 0
            detectedFacesCoords = self.faceDetector.detect(frame)
            for (x, y, w, h) in detectedFacesCoords:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 0, 0), 2)
            for gray_roi_img in self.faceDetector.get_cropped_faces():
                print(self.recognizer.predict(gray_roi_img))
        self.detectionFrame += 1
        return frame.tobytes()

class Container(BoxLayout):
    pass


#TODO: make the view from kv file

# Builder.load_string('''
# <CameraClick>:
#     size: root.size
#     orientation: 'vertical'
#     MyCamera:
#         id: camera
#         index: 0
#         resolution: (640, 480)
#         size_hint: 1, .7
#         play: True
#     Button:
#         text: 'Capture'
#         size_hint: 1, .3
#         on_press: root.capture()
# ''')


class FaceRecognitionCamera(App):
    def build(self):
        #TODO: add permission require

        return Container()


if __name__ == "__main__":
    FaceRecognitionCamera().run()