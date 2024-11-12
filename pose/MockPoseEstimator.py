from PoseEstimator import PoseEstimator
import tensorflow_hub as hub

class MockPoseEstimator(PoseEstimator):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.movenet = self.model.signatures['serving_default']

    def get_poses(self, video):
        print("Reached get_poses successfully")