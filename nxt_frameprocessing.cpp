//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
//# Please include the Github Repositories web URL if you are using this material.    #//
//#####################################################################################//
//#####################################################################################//
//#####################################################################################//
#include <tbb/tbb.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "all_header.h"


using namespace std;
using namespace cv;


void LANEDETECTION::nxt_frame(cuda::GpuMat& src, vector<float>&polyright_n, vector<float>&polyleft_n)
{


	vector<float>leftx;
	vector<float>lefty;
	vector<float>rightx;
	vector<float>righty;

	getIndicesOfNonZeroPixelsnext(src, leftx, lefty, rightx, righty);

	LANEDETECTION Polyfit;
	tbb::tbb_thread th1([&righty, &rightx, &polyright_n, &Polyfit]()
	{
		
		polyright_n = Polyfit.polyfiteigen(righty, rightx, 2);
	});

	tbb::tbb_thread th2([&lefty, &leftx, &polyleft_n, &Polyfit]()
	{
		polyleft_n = Polyfit.polyfiteigen(lefty, leftx, 2);
	});

	th1.join();
	th2.join();
}