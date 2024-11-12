from PoseEstimator import PoseEstimator
import cv2

class ProdPoseEstimator(PoseEstimator):
    def get_poses(self, video):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()

            self.draw_poses_for_frame(frame)
            
            cv2.imshow('Movenet Multipose', frame)
            
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()