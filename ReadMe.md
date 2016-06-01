# Robotics Lab Project
1. color tracking:
See folder [ColorCardDetection](ColorCardDetection); the purple paper is detected. The center is computed by the average coordinate of all purple pixels; the radius is estimated by the total area of purple pixels.

2. intrinsic camera calibration:
See folder [CameraCalibration](CameraCalibration); used to calibrate the camera and produce the xml settings file

3. extrinsic camera setup:
See folder [Playground](Playground); the produced camera calibration settings file is applied to calibrate the camera first, and the playground with solid black line, size 18 cm x 26 cm is detected and transformed. The center of purple color pixels is detected, computed, and tranformed to the playground coordinate system.  For robust detection, I implemented the paper `Dirk Farin, Susanne Krabbe, Wolfgang Effelsberg, Peter H. N. de With, "Robust Camera Calibration for Sport Videos using Court Models", SPIE Storage and Retrieval Methods and Applications for Multimedia, vol. 5307 p. 80-91, January 2004, San Jose (CA), USA`  ([pdf](http://www.dirk-farin.net/publications/data/Farin2004b.pdf), [website](http://www.dirk-farin.net/projects/sportsanalysis/index.html))

- Notes:
Only source code is included. The project is edited and compiled on the platform Windows 7, using Visual Studio 2010 C++ Express. The whole solution can be downloaded here: https://drive.google.com/folderview?id=0B7xdcc79qHBZMEZfSHZobk5ES3M&usp=sharing

Author: Fang-Lin He, 2014, May, 28.

If you have any question, please email me: fanglin.he@student.supsi.ch

Thank you!
