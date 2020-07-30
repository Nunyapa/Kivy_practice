import cv2


class FaceDetection:

    def __init__(self):
        self.cascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.last_detected_faces= []
        self.last_img = []

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascadeClassifier.detectMultiScale(gray, 1.3, 5)
        self.last_detected_faces = faces
        self.last_img = gray
        return faces

    def get_cropped_faces(self):
        list_cropped_faces = []
        for (x, y, w, h) in self.last_detected_faces:
            list_cropped_faces.append(self.last_img[y: y + h][x: x + w])
        return list_cropped_faces