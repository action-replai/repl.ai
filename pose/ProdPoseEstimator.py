from pose.PoseEstimator import PoseEstimator
import cv2

class ProdPoseEstimator(PoseEstimator):
    def get_poses(self, path):
        frame = cv2.imread(path)
        if frame is None:
            print("Error: Image not found.")
            return
        print(frame.shape)
        self.draw_poses_for_frame(frame)
        cv2.imshow("Pose Estimation", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cap = cv2.VideoCapture(video)
        # while cap.isOpened() and not self.stop_flag:
        #     ret, frame = cap.read()
        #     if not ret: continue

        #     print("frame shape:", frame.shape)

        #     self.draw_poses_for_frame(frame)
        #     cv2.imshow("write", frame)

        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break

        # cap.release()
        # cv2.destroyAllWindows()
        # self.stop_flag = False