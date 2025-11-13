import cv2
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

def viewFrames(vpath):
    # Open video file
    video = cv2.VideoCapture(vpath)
    frameMax = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frameStep = 1
    frame_idx = 1

    # Create figure
    fig = plt.figure(1)
    
    def update():
        if not (0 < frame_idx <= frameMax):
            return
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.clf()
            plt.imshow(frame)
            plt.title(f"Current Frame: {frame_idx}, Frame Step: {frameStep}")
            plt.draw()

    def on_key(event):
        nonlocal frame_idx, frameStep
        
        if event.key == 'up':
            frameStep += 1
            plt.title(f"Current Frame: {frame_idx}, Frame Step: {frameStep}")
            plt.draw()
            
        elif event.key == 'down':
            frameStep = max(frameStep - 1, 1)
            plt.title(f"Current Frame: {frame_idx}, Frame Step: {frameStep}")
            plt.draw()
            
        elif event.key == 'left':
            frame_idx = frame_idx - frameStep
            update()
            
        elif event.key == 'right':
            frame_idx = frame_idx + frameStep
            update()

    # Set up key press event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show initial frame
    update()
    plt.show()

    # Clean up
    video.release()
