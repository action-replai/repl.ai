from fastapi import FastAPI
from pose.ProdPoseEstimator import ProdPoseEstimator
from http.client import RemoteDisconnected
import signal
import threading
import time

app = FastAPI()
estimator = ProdPoseEstimator()
cancel_event = threading.Event()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/detect")
async def detect():
    cancel_event.clear()
    
    detection_thread = threading.Thread(target=run_pose_detection)
    detection_thread.start()

    return {"status": "Pose detection started"}

def run_pose_detection():
    try:
        estimator.get_poses(0, cancel_event)
    except RemoteDisconnected:
        print("Client disconnected")
    except Exception as e:
        print(f"Error during pose detection: {e}")
    finally:
        print("Pose detection finished")

def graceful_shutdown(signal, frame):
    print("Shutting down FastAPI server...")
    raise SystemExit

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)