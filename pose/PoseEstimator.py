from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import signal

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

        self.stop_flag = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, signum, frame):
        self.stop_flag = True

    @abstractmethod
    def get_poses(self, video): pass

    def get_keypoints(self, detections):
        # print(detections)
        stuff = detections['output_0'].numpy()[0,0,:,:]
        detections['output_0'].numpy()
        print(stuff)
        return stuff

    def draw_poses_for_frame(self, frame):
        input_img = tf.expand_dims(frame, axis=0)
        input_img = tf.image.resize_with_pad(input_img, self.input_size, self.input_size)
        input_img = tf.cast(input_img, dtype=tf.int32)

        # input_img_np = input_img.numpy().astype(np.uint8)[0]# Remove batch dimension
        # cv2_image = np.transpose(input_img_np, (1, 2, 0))
        # cv2_image = cv2.cvtColor(input_img_np, cv2.COLOR_BGR2RGB)
        # input_img_np = cv2.cvtColor(input_img_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Display with OpenCV
        # cv2.imshow("Input Image for Pose Estimation", input_img_np)
        # cv2.waitKey(0)  # Wait until any key is pressed
        # cv2.destroyAllWindows()

        print("input img shape:", input_img.shape)

        detections = self.model(input_img)
        keypoints_with_scores = self.get_keypoints(detections)

        self.draw(frame, keypoints_with_scores, self.EDGES, 0.1)

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        for edge in edges:
            p1, p2 = edge
            y1, x1, c1 = keypoints[p1]
            y2, x2, c2 = keypoints[p2]
            
            confident = (c1 > confidence_threshold) and (c2 > confidence_threshold)
            if confident:      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        for keypoint in keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

    def draw(self, frame, keypoints, edges, confidence_threshold):
        # for keypoint in keypoints:
            # print(keypoint)
        y, x, _ = frame.shape
        keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        self.draw_connections(frame, keypoints, edges, confidence_threshold)
        self.draw_keypoints(frame, keypoints, confidence_threshold)

    def rescale(self, frame):
        img = frame.copy()
        return tf.cast(img, dtype=tf.int32)
    
    