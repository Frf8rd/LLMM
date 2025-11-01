For starters, the task consists of working with video, for this we install the opencv-python library, in addition, in some cases there are mathematical things, for this we also install and import numpy.

1STEP: we create a function that can access the video by checking if it exists, then we find the total number of frames in the video

2STEP: we split the videos into frames at a step of 10 because of the video duration

3STEP: we transform the frames into binary form by the difference between the initial frame and a frame that we make gray, the background is black that's why we can easily use the function to find a contour for the object in the frame, then we calculate the area of ​​the received contour, we draw a rectangle on the object with the contour and then we find the center of the object

4STEP: we create the function to improve the contour first of all from the coordinate lists that we have we create 2 arrays that contain only x and y and we find these coordinates for a more stable and smooth contour

5STEP: we create the drawing function, first we create an empty image, and display the received coordinates on it

6STEP: we apply all the functions on each video by forming a forum and saving the images in a folder in the directory


##Pentru a porni progrmaa.

1.Crearea unui mediu virtual (venv)
python -m venv venv

2.Activare
venv\Scripts\activate

3.Instalarea librariilor necesare.
pip install -r requirements.txt
