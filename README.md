## Bits in the Wall with MoveNet
Steven Zhang, Patrick Gough, Alexander Tanimoto, Saatvik Vadlapatla

Submitted to BitCamp 2022, won Best Gamification Award

https://devpost.com/software/hole-in-the-wall-nwkf1e?ref_content=user-portfolio&ref_feature=in_progress

A PC game based on the hit worldwide tv show series "Hole in the Wall". It uses the body position capture ML model MoveNet.
## How to Run
https://drive.google.com/file/d/1NyHcpkJPyMBwHK4AI63AgLL5BQw57mkV/view?usp=sharing

download the portable folder with this link and extract it, then go into the folder named "final" and open final.exe

## Inspiration
We were inspired by "Hole in the Wall". In that game show, participants have to move their body into a certain position before a moving wall comes and knocks them down. While we can't have a giant moving wall like the game show, we liked the idea of a game where you have to move to a position before time runs out.

We think this project has exciting applications in fitness and exercise. It can get people to stretch and move around, or even do yoga poses! We think everybody that's been sitting in their chair and coding for a long time (maybe even at this hackathon) should use this app to get up and move.
## What it does
The object of the game is to fit your body inside the zone outlined in the blue area when the timer hits 0.

When you start the game, you are prompted to calibrate your body. Position your webcam so that it captures your whole body, then stand inside the zone for 3 seconds. After that, the game starts.

If your body's keypoints drawn by MoveNet collide with the blue area, the circle in the top right corner will turn red, and the colliding points will turn green. If you are in the zone when the timer hits 0, you pass and you get 1 point. Otherwise, you fail. After you fail 3 times, the game ends.
## How we built it
to display the game and it's screens, along with capturing video, we used python's OpenCV package. To predict the body skeleton, we used MoveNet. 
## Challenges we ran into
The first challenge we ran into was finding a good way to capture somebody's position. We were looking at semantic segmentation models at first, but they were slower. After finding MoveNet, we decided to pivot to using it instead.
The next large challenge was getting the python script we had to run on all of our different machines. Some of us had apple computers, and we had to spend a few hours getting the correct python, tensorflow, and opencv versions.
The next challenge was understanding the numpy array format that images were dealt with in our source code, and finding methods to do the tasks we needed to detect collision with our mask, as well as creating different while loops for different parts of the game.

Another challenge was attempting (unsuccessfully) to host a web version of this application.  When we first tested the model, we made a small Django site to relay a user's webcam feed to a local server running the model.  Excited, we sought to host it on a DigitalOcean box so we could have 24/7 runtime.  Unfortunately, although we did not run into any installation problems with the box, the specs were insufficient to run TensorFlow, so we tried a raspberry pi.  After hours trying to install a version of Python that supports TensorFlow, we realized that the pi's ARM processor was also insufficient.  We decided to open an ssh tunnel to one of our laptops instead.  As we developed the rest of the application, we tried to port the entire game to a website, but the site was very laggy.  We tried using WebSockets to stream a constant game feed from the server, but unfortunately that did not work.
## Accomplishments that we're proud of
Creating the game
## What we learned
We learned how OpenCV draws pictures and how pictures are stored in numpy arrays. We also learned how to switch between python versions and install packages, and how to collaborate using GitHub.
## What's next for Hole in the wall
If we were to expand on this project, here are some things we thought of adding:
1. difficulty levels
2. multiplayer
3. leaderboard system
4. keypoint target positions you have to hit

# Credits
Nicholas Renotte: https://www.youtube.com/watch?v=SSW9LzOJSus

MoveNet: https://tfhub.dev/google/movenet/singlepose/lightning/4
