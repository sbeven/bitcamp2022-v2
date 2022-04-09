#!/usr/bin/env python
# coding: utf-8

# # 0. Install and Import Dependencies

# In[ ]:


# pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python matplotlib

# In[ ]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2



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
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def draw_keypoints(frame, shaped, confidence_threshold, colors):
    for i in range(0, len(shaped)):
        ky, kx, kp_conf = shaped[i]
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, colors[i], -1)
            # # 4. Draw Edges


# In[ ]:
# In[ ]:


# # 1. Load Model

# In[ ]:


interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()


# # 2. Make Detections

# In[ ]:

mask_path = "Drawing.sketchpad.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
colors = [(255, 255, 255)] * 17
cap = cv2.VideoCapture(0)
collision = False
while cap.isOpened():
    collision = False
    ret, frame = cap.read()

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

    frame = cv2.resize(frame, dsize=(843, 640), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, dsize=(843, 640), interpolation=cv2.INTER_CUBIC)

    # change dtype of mask
    mask = mask.astype('uint8')

    frame = np.concatenate((frame, np.full((640, 843, 1), 255, dtype=np.uint8)), axis=2)
    frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    # collisions
    for i in range(0, len(shaped)):
        xpos = shaped[i][0]
        ypos = shaped[i][1]
        if xpos < y and ypos < x:
            if mask[int(xpos), int(ypos), 3] > 0:
                colors[i] = (255, 0, 0)
                Collision = True;
            else:
                colors[i] = (255, 255, 255)
    # Rendering
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, shaped, 0.4, colors)

    frame = cv2.flip(frame, 1)

    cv2.imshow('MoveNet Lightning', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('MoveNet Lightning', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()


# # 3. Draw Keypoints

# In[ ]:






# In[ ]:






