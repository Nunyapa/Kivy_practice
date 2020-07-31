import cv2
import pickle
from kivy.core.audio import SoundLoader
from FaceDetector import FaceDetector


class FaceRecognizer():

    def __init__(self):
        self.SOUND_PLAY_AFTER = 50
        self.sounds = {}
        self.detector = FaceDetector("haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(2, 8)
        self.recognizer.read('trained.yml')
        self.labels = {}
        with open("labels.pickle", 'rb') as f:
            labels = pickle.load(f)
            self.labels = {v: k for k, v in labels.items()}
        for k in labels:
            sound = SoundLoader.load(f'data/Sounds/{k}.wav')
            sound.bind(on_stop=self.itDone)
            self.sounds[k] = sound

        self.counter_frame = 0
        self.person_scores = {k: 0 for k in labels}
        self.last_person = -1

    def detect(self, img):
        '''inp image is always should be  rgb img'''

        faces = self.detector.detect(img)
        return faces

    def draw(self, img, detectedFacesCoords, labels, confs):
        counter = 0
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        color = (255,255,255)
        stroke = 2
        for (x, y, w, h) in detectedFacesCoords:
            x1 = x + w
            y1 = y + h
            cv2.rectangle(img, (x, y), (x1, y1), (30, 150, 40), 3)
            name = f'{labels[counter]}({confs[counter]})'
            cv2.putText(img, name, (x, y + 5), font, 1, color, stroke, cv2.LINE_AA)
            counter += 1
        return img

        
    
    def predict(self, img):
        self.counter_frame += 1
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_coords = self.detect(gray_img)
        img_array = [gray_img[y: y + h, x: x + w] for x, y, w, h in faces_coords]
        labels = []
        confs = []
        for gray_roi_img in img_array:
            try:
                id_, conf = self.recognizer.predict(gray_roi_img)
                if conf >= 90:
                    labels.append(self.labels[id_])
                    confs.append(str(conf)[:3])
                    self.person_scores[self.labels[id_]] += 1
                else:
                    labels.append('unknown')
                    confs.append(0)
            except:
                pass



        if self.counter_frame > self.SOUND_PLAY_AFTER:
            cur_person = max(self.person_scores.items(), key=lambda i: i[1])[0]
            if self.person_scores[cur_person] < 10:
                cur_person = 'unknown'
            if self.last_person != cur_person and cur_person != 'unknown':
                print("play")
                self.sounds[cur_person].play()
            self.last_person = cur_person
            self.counter_frame = 0
            self.person_scores = {k: 0 for k in self.person_scores}

        img = self.draw(img, faces_coords, labels, confs)
        return img


    def itDone(self, sound):
        sound.seek(0)

