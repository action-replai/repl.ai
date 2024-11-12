from ProdPoseEstimator import ProdPoseEstimator
from MockPoseEstimator import MockPoseEstimator

video = "../videos/IMG_6366.mov"

estimator = ProdPoseEstimator()
estimator.get_poses(video)