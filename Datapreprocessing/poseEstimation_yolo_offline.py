from ultralytics import YOLO
import cv2 as cv
import time
import csv
import os
from pathlib import Path
import toml

# hard-coded joint pair
KEYPOINT_DICT={
    "nose":0,
    "left_eye":1,
    "left_ear":2,
    "right_eye":3,
    "right_ear":4,
    "left_shoulder":5,
    "right_shoulder":6,
    "left_elbow":7,
    "right_elbow":8,
    "left_wrist":9,
    "right_wrist":10,
    "left_hip":11,
    "right_hip":12,
    "left_knee":13,
    "right_knee":14,
    "left_ankle":15,
    "right_ankle":16
}

KEYPOINT_PAIR = [(KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["left_elbow"]),
                 (KEYPOINT_DICT["left_elbow"],KEYPOINT_DICT["left_wrist"]),
                 (KEYPOINT_DICT["right_shoulder"],KEYPOINT_DICT["right_elbow"]),
                 (KEYPOINT_DICT["right_elbow"],KEYPOINT_DICT["right_wrist"]),
                 (KEYPOINT_DICT["left_hip"],KEYPOINT_DICT["left_knee"]),
                 (KEYPOINT_DICT["left_knee"],KEYPOINT_DICT["left_ankle"]),
                 (KEYPOINT_DICT["right_hip"],KEYPOINT_DICT["right_knee"]),
                 (KEYPOINT_DICT["right_knee"],KEYPOINT_DICT["right_ankle"]),
                 (KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["right_shoulder"]),
                 (KEYPOINT_DICT["left_hip"],KEYPOINT_DICT["right_hip"]),
                 (KEYPOINT_DICT["left_shoulder"],KEYPOINT_DICT["left_hip"]),
                 (KEYPOINT_DICT["right_shoulder"],KEYPOINT_DICT["right_hip"])]


# Load a model
model = YOLO('Datapreprocessing\\yolov8n-pose.pt')  # build from YAML and transfer weights

# Read project_name from config/app_config.toml and build paths under ./Data/{project_name}
project_root = Path(__file__).resolve().parent.parent  # .../Project Code
cfg_path = project_root / 'config' / 'config.toml'
cfg_data = toml.load(str(cfg_path))
project_name = cfg_data['project']['project_name']

# used for recording
fcount = 0
t_prev = 0
FPS = 60            # recording frame rate
TIME_LAPSE = 1/FPS
THRESHOLD = 0.75

video_path = project_root / 'Data' / project_name / 'videos' / 'cam01.mp4'
cap = cv.VideoCapture(str(video_path))
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)

if not cap.isOpened():
    print("Cannot open file")
    exit()

out = []
while True:
    print(fcount)
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    results = model(frame, show=False, verbose=False)
    keypoints= results[0].keypoints.xy      # all keypoints in original pixel coordinate for each detected person
    score = results[0].keypoints.conf
    
    # no person detected
    if (keypoints is None) or (score is None):
        continue
    
    for person,conf in zip(keypoints,score):
        kps = person.tolist()
        val  = conf.tolist()
        # keypoints visualization 
        # for pair in KEYPOINT_PAIR:
        #     try:
        #         if val[pair[0]]<THRESHOLD or val[pair[1]]<THRESHOLD:
        #             continue
        #         pt1 = tuple(int(i) for i in kps[pair[0]])
        #         pt2 = tuple(int(i) for i in kps[pair[1]])
        #         frame = cv.circle(frame, pt1, 6, (255,255,255) , -1) #(BGR)
        #         frame = cv.circle(frame, pt2, 6, (255,255,255) , -1)
        #         frame = cv.line(frame, pt1, pt2, (0,255,0), 3)
        #     except:
        #         continue

        #only output right_elbow-right_wrist
        # right_shoulder = list(int(i) for i in kps[KEYPOINT_DICT["right_shoulder"]])
        # right_elbow = list(int(i) for i in kps[KEYPOINT_DICT["right_elbow"]])
        # right_wrist = list(int(i) for i in kps[KEYPOINT_DICT["right_wrist"]])
        # right_shoulder_conf = val[KEYPOINT_DICT["right_shoulder"]]
        # right_elblw_conf = val[KEYPOINT_DICT["right_elbow"]]
        # right_wrist_conf = val[KEYPOINT_DICT["right_wrist"]]
        left_shoulder = list(int(i) for i in kps[KEYPOINT_DICT["left_shoulder"]])
        left_elbow = list(int(i) for i in kps[KEYPOINT_DICT["left_elbow"]])
        left_wrist = list(int(i) for i in kps[KEYPOINT_DICT["left_wrist"]])
        left_shoulder_conf = val[KEYPOINT_DICT["left_shoulder"]]
        left_elblw_conf = val[KEYPOINT_DICT["left_elbow"]]
        left_wrist_conf = val[KEYPOINT_DICT["left_wrist"]]
        
        # add to original frame for visualization
        #frame = cv.circle(frame, right_elbow, 6, (255,255,255) , -1) #(BGR)
        #frame = cv.circle(frame, right_wrist, 6, (255,255,255) , -1)
        #frame = cv.line(frame, right_elbow, right_wrist, (0,255,0), 3)
        
        out.append([fcount]+left_shoulder+[left_shoulder_conf]+left_elbow+[left_elblw_conf]+left_wrist+[left_wrist_conf])

    #cv.imwrite(f"./{fcount}.jpg",frame)
    fcount+=1

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# Write measurement CSV into ./Data/{project_name}/measurement_data
measurement_dir = project_root / 'Data' / project_name / 'measurement_data'
os.makedirs(measurement_dir, exist_ok=True)
with open(str(measurement_dir / 'measurement.csv'), "w", newline="") as f:
    header = ['index', 'lshoulder u','lshoulder v', 'lshoulder conf','lelbow u', 'lelbow v', 'lelbow conf','lwrist u','lwrist v','lwrist conf']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(out)
    f.close()







