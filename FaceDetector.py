import cv2


class FaceDetector:

    def __init__(self, filename):
        self.cascadeClassifier = cv2.CascadeClassifier(filename)
        # self.last_detected_faces= []
        # self.last_img = []

    def detect(self, img):
        '''waiting for gray img'''

        faces = self.cascadeClassifier.detectMultiScale(img, 1.3, 5)
        return faces