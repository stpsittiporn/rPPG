import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import OrderedDict


class Face_utilities():

    def __init__(self, face_width=200):
        self.detector = None
        self.predictor = None
        self.age_net = None
        self.gender_net = None

    def face_detection(self, frame):

        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        return rects

    def get_landmarks(self, frame):
        if self.predictor is None:
            #print("[INFO] load " + type + " facial landmarks model ...")
            self.predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
            print("[INFO] Load model - DONE!")

        if frame is None:
            return None, None

        # face must be gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.face_detection(frame)

        if len(rects) < 0 or len(rects) == 0:
            return None, None

        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        return shape, rects

    def no_age_gender_face_process(self, frame):

        shape, rects = self.get_landmarks(frame)
        if shape is None:
            return None

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        face = frame[y:y + h, x:x + w]
        return rects, face, shape










