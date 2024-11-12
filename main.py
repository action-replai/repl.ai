from fastapi import FastAPI
from pose import ProdPoseEstimator, MockPoseEstimator

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/detect")
async def detect():
    # estimator = ProdPoseEstimator()
    estimator = MockPoseEstimator()
    estimator.get_poses()