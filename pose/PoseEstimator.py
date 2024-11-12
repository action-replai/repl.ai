from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

class PoseEstimator(ABC):
    def __init__(self):
        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        self.model = self.module.signatures['serving_default']
        self.EDGES = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), 
            (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), 
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

    @abstractmethod
    def get_poses(self, video): pass

    def get_keypoints(self, detections):
        # print(detections)
        return detections['output_0'].numpy().squeeze()

    def draw_poses_for_frame(self, frame):
        input_img = tf.expand_dims(frame, axis=0)
        input_img = tf.image.resize_with_pad(input_img, self.input_size, self.input_size)
        input_img = tf.cast(input_img, dtype=tf.int32)

        detections = self.model(input_img)
        print(f"detections: {detections}")
        keypoints_with_scores = self.get_keypoints(detections)

        self.draw(frame, keypoints_with_scores, self.EDGES, 0.1)

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        frame_mapped_keypoints = np.multiply(keypoints, [y,x,1])
        frame_mapped_keypoints = np.squeeze(frame_mapped_keypoints)
        
        for edge in edges:
            p1, p2 = edge
            print(keypoints[p1])
            y1, x1, c1 = keypoints[p1]
            y2, x2, c2 = keypoints[p2]
            
            confident = (c1 > confidence_threshold) and (c2 > confidence_threshold)
            if confident:      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        frame_mapped_keypoints = np.multiply(keypoints, [y,x,1])
        frame_mapped_keypoints = np.squeeze(frame_mapped_keypoints)
        
        for keypoint in frame_mapped_keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

    def draw(self, frame, keypoints, edges, confidence_threshold):
        for keypoint in keypoints:
            self.draw_connections(frame, keypoint, edges, confidence_threshold)
            self.draw_keypoints(frame, keypoint, confidence_threshold)

    def rescale(self, frame):
        img = frame.copy()
        return tf.cast(img, dtype=tf.int32)
    