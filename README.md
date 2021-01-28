# Advanced Lane Finding Project and Yolo Objection Detection.   
## Tabel of Content
1 - [Algorithm Details](#algorithm-details).  
2 - [YOLO Object Detection](#yolo5s-object-detection).  
3 - [YOLO5S Detection Setup](#yolo5s-detection-setup).  
4 - [Dependencies and Compiling](#dependencies-and-compiling).  
## Algorithm Details.     
### Distortion corrected calibration image.   
The code for this step is contained in the calibration.cpp [Here](https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/main/calibration.cpp).  
Start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here we are assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Then use the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the calibrateCamera() function.  
Example: 
<p align="center">
  <img src="images/image1.png">
</p>  


#### Pipeline.   
At First,  resize the image and then  convert frame as Bird view and then use a combination of color and gradient thresholds to generate a binary image.  
**Step 1:** Undisort Image.   
**Step 2:** Binary Image.   
**Step 3:** Take a histogram along all the columns in the lower half of the image and split histogram for two sides for each lane.   
**Step 4:** Use the two highest peaks from histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image to determine where the lane lines go.   
Example: 
<p align="center">
  <img width="640" height="640" src="images/image2.png">
</p>  


**Step 5:**  Identify lane-line pixels.  
Find all non zero pixels.  
Example:   
<p align="center">
  <img width="640" height="360" src="images/image3.png">
</p> 


**Step 6:** Fit their positions with a polynomial.   
After performing 2nd order polynomial fit for nonzero pixels, drawing polyline and unwrap image the final output.  
Example:   
<p align="center">
  <img width="640" height="360" src="images/image4.png">
</p> 


### Radius of curvature of the lane and the position of the vehicle with respect to center. 
Get the left and right cordinates and calculate the midpoint of lanes and use the image center as reference to calculate distance away from center.  
1-	laneyolo::get_centerdistance() – Distance of Vechicle from center.  
2-	laneyolo::get_leftcurvature() – Left Lane Curvature.  
3-	laneyolo::get_rightcurvature() – Right Lane Curvature.  


### YOLO5s Object Detection.  
The Pytorch Model is retrained on partial COCO dataset for 24 classes More Information [Here]( https://github.com/ultralytics/yolov5).  
The detail for trained classes can be found [Here]( extras/classes). The implementation is parallel, the preprocessing is done on one thread and detection and post processing is done on another and Lane detection on another thread.  
Example:   
<p align="center">
  <img width="640" height="360" src="images/image6.png">
</p> 

## YOLO5S Detection Setup.  
1-  laneyolo::set_batchsize(1) sets batch size for detection. Higher the no higher FPS. Default is 1.    
2-  laneyolo::set_confthres(0.6) sets Confidance Threshold for detection. Default is 0.6.  
3-  laneyolo::set_iouthres(0.4) sets IOU Threshold for detection. Default is 0.4.  

## Testing result 21 FPS with batch size 1 and Lane detection 21 FPS parallel (RTX 2080, Intel i7).  


## Dependencies and Compiling. 
### Environment Windows 10.  
1-	Visual Studio 2019.  
2-	CUDA 11.1. For Windows installation Guide and Requirements [Here]( https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software).     
3-	Pytorch 1.9.0. Pytorch C++ library on Windows [Here]( https://pytorch.org/get-started/locally/).    
4-	OpenCV 4.5 or Greater. More information can be found [Here]( https://jamesbowley.co.uk/accelerate-opencv-4-5-0-on-windows-build-with-cuda-and-python-bindings/).    
5-	Pytorch trained model (best.torchscript.pt) is included in repository.  
6-  Eigen Library.  
7-  CMake.  


## Extra info to Build.
1-  The Windows bat file is [Here](extras/ApplicationBuilder.bat) to build.  
2-  Change EIGEN3_DIR variable in bat file as per your Eigen install location.  
3-  Change TORCH_DIR variable in bat file as per your libtorch install location.  
4-  OPENCV bin path needs to be added to ENV path.  


# For Linux Tensorflow YOLO Version Click [Here](https://github.com/apreddyy/LaneandYolovehicle-DetectionLinux)  
# Thanks
