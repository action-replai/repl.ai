from PoseEstimator import PoseEstimator
import cv2

class ProdPoseEstimator(PoseEstimator):
    def get_poses(self, video):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()

            input_img = self.rescale(frame)
            
            results = self.movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
            
            self.apply_pose_to_people(frame, keypoints_with_scores, self.EDGES, 0.1)
            
            cv2.imshow('Movenet Multipose', frame)
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break