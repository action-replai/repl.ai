from ab import ABC, abstractmethod
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

class PoseEstimator(ABC):
    def __init__(self):
        self.model = hub.load('http://tfhub.dev/google/movement/multipose/lightning/1')
        self.movenet = self.model.signatures['seriving_default']
        self.EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), 
            (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), 
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    @abstractmethod
    def get_poses(self, video): pass


    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        rescaled_points = np.multiply(keypoints, [y,x,1])
        shaped = np.squeeze(rescaled_points)
        
        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            confident = (c1 > confidence_threshold) and (c2 > confidence_threshold)
            if confident:      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

    def apply_pose_to_people(self, frame, keypoints, edges, confidence_threshold):
        for person in keypoints:
            self.draw_connections(frame, person, edges, confidence_threshold)
            self.draw_keypoints(frame, person, confidence_threshold)

    def rescale(self, frame):
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        return tf.cast(img, dtype=tf.int32)
    