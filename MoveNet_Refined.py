#import os
#os.system("pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python matplotlib")

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import time
import random


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


def draw_keypoints(frame, shaped, confidence_threshold, colors):
    for i in range(0, len(shaped)):
        ky, kx, kp_conf = shaped[i]
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 10, colors[i], -1)




interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# initializations
paths = ["Mask1.png",
         "Mask2.png",
         "Mask3.png",
         "Mask4.png",
         "Mask5.png",
         "Mask6.png",
         "Mask7.png",
         "Mask8.png",
         "Mask9.png",
         "Mask10.png",
         "Mask11png",
         ]
name = "Bits in the Wall"
videoWidth = 750
videoHeight = 750
colors = [(255, 255, 255)] * 17
cap = cv2.VideoCapture(0)
collision = False
# initialize timer
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
fontcolor = (0, 0, 0)
fails = 0
score = 0
rounds = 0
end = False
skipEvent = False
while not end: # Repeat indefinitely as long not "q"
    fails = 0
    score = 0
    rounds = 0
    skipEvent = False
    cut = False # Stop the game and show exit screen
    restart = False # Allow user to restart in-game

    # user calibration
    start = time.time()
    finish = time.time()
    skipEvent = False
    mask_path = paths[0]  # replace with calibration mask
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    while not end and not skipEvent and not cut and not restart:

        collision = False
        ret, frame = cap.read()
        # we need a square image to pass to the model
        frame = frame[:,80:560]
        # Reshape image
        img = frame.copy()
        img = np.expand_dims(img, axis=0)
        img = tf.image.resize_with_pad(img, 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)
        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # resize and render mask

        frame = cv2.resize(frame, dsize=(videoWidth, videoHeight), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=(videoWidth, videoHeight), interpolation=cv2.INTER_CUBIC)

        # change dtype of mask
        mask = mask.astype('uint8')

        frame = np.concatenate((frame, np.full((videoHeight, videoWidth, 1), 255, dtype=np.uint8)), axis=2)
        frame = cv2.addWeighted(frame, 1, mask, 0.8, 0)

        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
        # collisions
        for i in range(0, len(shaped)):
            xpos = shaped[i][0]
            ypos = shaped[i][1]
            confidence = shaped[i][2]
            if xpos < y and ypos < x and confidence > 0.4:
                if mask[int(xpos), int(ypos), 3] > 0:
                    colors[i] = (0, 0, 255)
                    collision = True;
                else:
                    colors[i] = (255, 255, 255)
        # Rendering
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, shaped, 0.4, colors)

        frame = cv2.flip(frame, 1)
        # put the timer on the screen
        if collision:
            start = time.time()
            frame = cv2.putText(frame, "Calibrate body", (15, 50), \
                                fontface, fontscale, fontcolor, thickness=3)
            # red dot to show that there's a collision
            frame = cv2.circle(frame, (videoWidth - 25, 25), 20, (0 ,0 ,255), thickness=-1)
        else:
            frame = cv2.putText(frame, "Hold position: " + str(3 - int(finish - start)), (15, 50), \
                                fontface, fontscale, fontcolor, thickness=3)
            frame = cv2.circle(frame, (videoWidth - 25, 25), 20, (0, 255, 0), thickness=-1)
        cv2.imshow(name, frame)
        input = cv2.waitKey(10)
        if input == ord('q'):
            end = True
        if input == ord('s'):
            skipEvent = True
        if input == ord('c'):
            cut = True
        if input == ord('r'):
            restart = True
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            end = True
        finish = time.time()
        if finish - start > 3:
            skipEvent = True
    skipEvent = False
    # game start text
    start = time.time()
    finish = time.time()
    while finish - start < 3 and not end and not skipEvent and not cut and not restart:
        frame = cv2.rectangle(frame, (0, 0), (videoWidth, videoHeight), (255, 255, 255), thickness=-1)
        frame = cv2.putText(frame, "Game will start in " + \
                            str(3 - int(finish - start)), (int(videoWidth / 2) - 330 ,int(videoHeight / 2)), \
                            fontface, fontscale, fontcolor, thickness=3)
        cv2.imshow(name, frame)
        finish = time.time()
        input = cv2.waitKey(10)
        if input == ord('q'):
            end = True
        if input == ord('s'):
            skipEvent = True
        if input == ord('c'):
            cut = True
        if input == ord('r'):
            restart = True
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            end = True

    random.shuffle(paths)
    used = []
    skipEvent = False
    # game start screen
    while fails < 3 and not end and not cut and not restart:

        start = time.time()
        finish = time.time()
        skipEvent = False
        mask_path = paths.pop()
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        used.append(mask_path)

        # game loop
        while finish - start < 10 and not end and not skipEvent and not cut and not restart:
            collision = False
            ret, frame = cap.read()
            frame = frame[:, 80:560]
            # Reshape image
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)
            # Setup input and output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Make predictions
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

            # resize and render mask

            frame = cv2.resize(frame, dsize=(videoWidth, videoHeight), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, dsize=(videoWidth, videoHeight), interpolation=cv2.INTER_CUBIC)

            # change dtype of mask
            mask = mask.astype('uint8')

            frame = np.concatenate((frame, np.full((videoHeight, videoWidth, 1), 255, dtype=np.uint8)), axis=2)
            frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)


            y, x, c = frame.shape
            shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
            # collisions
            for i in range(0, len(shaped)):
                xpos = shaped[i][0]
                ypos = shaped[i][1]
                confidence = shaped[i][2]
                if xpos < y and ypos < x and confidence > 0.4:
                    if mask[int(xpos), int(ypos), 3] > 0:
                        colors[i] = (0, 0, 255)
                        collision = True;
                    else:
                        colors[i] = (255, 255, 255)
            # Rendering
            draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
            draw_keypoints(frame, shaped, 0.4, colors)

            frame = cv2.flip(frame, 1)
            # put the timer on the screen
            frame = cv2.putText(frame, str(10 - int(finish - start)), (15, 50), \
                                fontface, fontscale, fontcolor, thickness=3)
            if collision:
                # show collision circle
                frame = cv2.circle(frame, (videoWidth - 25, 25), 20, (0, 0, 255), thickness=-1)
            else:
                frame = cv2.circle(frame, (videoWidth - 25, 25), 20, (0, 255, 0), thickness=-1)
            cv2.imshow(name, frame)
            input = cv2.waitKey(10)
            if input == ord('q'):
                end = True
            if input == ord('s'):
                skipEvent = True
            if input == ord('c'):
                cut = True
            if input == ord('r'):
                restart = True
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                end = True
            finish = time.time()
        skipEvent = False

        # display a fail/success screen
        start = time.time()
        finish = time.time()
        if collision:
            fails += 1
        else:
            score += 1
        rounds += 1
        while finish - start < 3 and not end and not skipEvent and not cut and not restart:
            if fails == 3:
                frame = cv2.rectangle(frame, (0, 0), (videoWidth, videoHeight), (0, 0, 255), thickness=-1)
                frame = cv2.putText(frame, "Game Over", (int(videoWidth / 2) - 180 ,int(videoHeight / 2)), \
                                    fontface, fontscale, fontcolor, thickness=3)
                frame = cv2.putText(frame, str(3 - int(finish - start)), (15, 50), \
                                    fontface, fontscale, fontcolor, thickness=3)
            elif collision:
                frame = cv2.rectangle(frame, (0, 0), (videoWidth, videoHeight), (0, 0 ,255), thickness = -1)
                frame = cv2.putText(frame, "Fail", (int(videoWidth / 2) - 60 ,int(videoHeight / 2)), \
                                    fontface, fontscale, fontcolor, thickness=3)
                frame = cv2.putText(frame, str(3 - int(finish - start)), (15, 50), \
                                    fontface, fontscale, fontcolor, thickness=3)
            else:
                frame = cv2.rectangle(frame, (0, 0), (videoWidth, videoHeight), (0, 255, 0), thickness=-1)
                frame = cv2.putText(frame, "Pass", (int(videoWidth / 2) - 80 ,int(videoHeight / 2)), \
                                    fontface, fontscale, fontcolor, thickness=3)
                frame = cv2.putText(frame, str(3 - int(finish - start)), (15, 50), \
                                    fontface, fontscale, fontcolor, thickness=3)
            cv2.imshow(name, frame)
            finish = time.time()
            input = cv2.waitKey(10)
            if input == ord('q'):
                end = True
            if input == ord('s'):
                skipEvent = True
            if input == ord('c'):
                cut = True
            if input == ord('r'):
                restart = True
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                end = True

        # Recycle paths if all images used
        if len(paths) == 0:
            while len(used) > 0:
                paths.append(used.pop())
            random.shuffle(paths)

    # end screen
    while not end and not restart:
        frame = cv2.rectangle(frame, (0, 0), (videoWidth, videoHeight), (255, 255, 255), thickness=-1)
        frame = cv2.putText(frame, "Rounds Survived: " + str(rounds), (int(videoWidth / 2) - 300, int(videoHeight / 2) - 100), \
                            fontface, \
                            fontscale, fontcolor, thickness=3)
        frame = cv2.putText(frame, "Score: " + str(score), (int(videoWidth / 2) - 150, int(videoHeight / 2) - 40), fontface, \
                            fontscale, fontcolor, thickness=3)
        frame = cv2.putText(frame, "Press q to quit", (int(videoWidth / 2) - 250, int(videoHeight / 2) + 20), fontface, \
                            fontscale, fontcolor, thickness=3)
        frame = cv2.putText(frame, "Press r to restart", (int(videoWidth / 2) - 300, int(videoHeight / 2) + 80), fontface, \
                            fontscale, fontcolor, thickness=3)
        cv2.imshow(name, frame)
        input = cv2.waitKey(10)
        if input == ord('c') or input == ord('q'):
            end = True
        if input == ord('r'):
            restart = True
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            end = True

    if restart:
        while len(used) > 0:
            paths.append(used.pop())
        paths.sort()

cap.release()
cv2.destroyAllWindows()
