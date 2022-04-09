# Hole in the Wall with MoveNet
Steven Zhang, Patrick Gough, Alexander Tanimoto, Saatvik Vadlapatla
A PC game based on the hit worldwide tv show series "Hole in the Wall". It uses the body position capture ML model MoveNet.
# Instructions
The object of the game is to fit your body inside the zone outlined in the blue area when the timer hits 0. 

When you start the game, you are prompted to calibrate your body. Position your webcam so that it captures your whole body, then stand inside the zone for 3 seconds.
After that, the game starts.

If your body's keypoints drawn by MoveNet collide with the 
blue area, the circle in the top right corner will turn red, and the colliding points will turn green. If you are in the zone when the timer hits 0, you pass and
you get 1 point. Otherwise, you fail. After you fail 3 times, the game ends. 

important keys:
s - skip section
q- quit


https://www.youtube.com/watch?v=SSW9LzOJSus
