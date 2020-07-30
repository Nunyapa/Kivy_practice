import cv2


class FaceRecognizer():

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trained.yml')

    def predict(self, gray_roi_img):
        if len(gray_roi_img) > 0:

            id_, conf = self.recognizer.predict(gray_roi_img)
            if conf >= 55 and conf <= 100:
                return id_
        return -1