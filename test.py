import requests
from pose.ProdPoseEstimator import ProdPoseEstimator

# url = "http://127.0.0.1:8000"

# try:
#     response = requests.get(f"{url}/detect")
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")


# video = "../videos/IMG_6366.mov"
path = "data/IMG_5913.JPG"

estimator = ProdPoseEstimator()
estimator.get_poses(path)