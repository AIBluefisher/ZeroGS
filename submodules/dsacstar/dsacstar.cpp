/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac

Copyright (c) 2016, TU Dresden
Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/extension.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "thread_rand.h"
#include "stop_watch.h"

#include "dsacstar_types.h"
#include "dsacstar_util.h"
//#include "dsacstar_util_rgbd.h"
#include "dsacstar_loss.h"
#include "dsacstar_derivative.h"

#define MAX_REF_STEPS 100 // max pose refienment iterations
#define MAX_HYPOTHESES_TRIES 1000000 // repeat sampling x times hypothesis if hypothesis is invalid

/**
 * @brief Estimate a camera pose based on a scene coordinate prediction
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param outPoseSrc Camera pose (output parameter), (4x4) tensor containing the homogeneous camera tranformation matrix.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param maxReproj Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 * @return The number of inliers for the output pose.
 */
int dsacstar_rgb_forward(
	at::Tensor sceneCoordinatesSrc, 
	at::Tensor outPoseSrc,
	int ransacHypotheses, 
	float inlierThreshold,
	float fx,
	float fy,
	float ppointX,
	float ppointY,
	float inlierAlpha,
	float maxReproj,
	int subSampling,
	int maxHypothesesTries = 10000000,
	bool verbose = false)
{
	ThreadRand::init();

	// access to tensor objects
	dsacstar::coord_t sceneCoordinates = 
		sceneCoordinatesSrc.accessor<float, 4>();

	// dimensions of scene coordinate predictions
	int imH = sceneCoordinates.size(2);
	int imW = sceneCoordinates.size(3);

	// internal camera calibration matrix
	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
	camMat(0, 0) = fx;
	camMat(1, 1) = fy;
	camMat(0, 2) = ppointX;
	camMat(1, 2) = ppointY;	

	// calculate original image position for each scene coordinate prediction
	cv::Mat_<cv::Point2i> sampling = 
		dsacstar::createSampling(imW, imH, subSampling, 0, 0);

	if (verbose) {
		std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
	}
	
	StopWatch stopW;

	// sample RANSAC hypotheses
	std::vector<dsacstar::pose_t> hypotheses;
	std::vector<std::vector<cv::Point2i>> sampledPoints;  
	std::vector<std::vector<cv::Point2f>> imgPts;
	std::vector<std::vector<cv::Point3f>> objPts;

	dsacstar::sampleHypotheses(
		sceneCoordinates,
		sampling,
		camMat,
		ransacHypotheses,
		maxHypothesesTries,
		inlierThreshold,
		hypotheses,
		sampledPoints,
		imgPts,
		objPts);

	if (verbose) {
		std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
		std::cout << BLUETEXT("Calculating scores.") << std::endl;
	}
    
	// compute reprojection error images
	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
	cv::Mat_<double> jacobeanDummy;

	#pragma omp parallel for 
	for(unsigned h = 0; h < hypotheses.size(); h++)
    	reproErrs[h] = dsacstar::getReproErrs(
		sceneCoordinates,
		hypotheses[h], 
		sampling, 
		camMat,
		maxReproj,
		jacobeanDummy);

    // soft inlier counting
	std::vector<double> scores = dsacstar::getHypScores(
    	reproErrs,
    	inlierThreshold,
    	inlierAlpha);

	if (verbose) {
		std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
		std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;
	}

	// apply soft max to scores to get a distribution
	std::vector<double> hypProbs = dsacstar::softMax(scores);
	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
	int hypIdx = dsacstar::draw(hypProbs, false); // select winning hypothesis
	
	if (verbose) {
		std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl; 
		std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;

		std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
		std::cout << BLUETEXT("Refining winning pose:") << std::endl;
	}

	// refine selected hypothesis
	cv::Mat_<int> inlierMap;

	dsacstar::refineHyp(
		sceneCoordinates,
		reproErrs[hypIdx],
		sampling,
		camMat,
		inlierThreshold,
		MAX_REF_STEPS,
		maxReproj,
		hypotheses[hypIdx],
		inlierMap);

	if (verbose) {
		std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	}

	// write result back to PyTorch
	dsacstar::trans_t estTrans = dsacstar::pose2trans(hypotheses[hypIdx]);

	auto outPose = outPoseSrc.accessor<float, 2>();
	for(unsigned x = 0; x < 4; x++)
	for(unsigned y = 0; y < 4; y++)
		outPose[y][x] = estTrans(y, x);

	// Return the inlier count. cv::sum returns a scalar, so we return its first element.
	return cv::sum(inlierMap)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_rgb", &dsacstar_rgb_forward, "DSAC* forward (RGB)");
//	m.def("backward_rgb", &dsacstar_rgb_backward, "DSAC* backward (RGB)");
//	m.def("forward_rgbd", &dsacstar_rgbd_forward, "DSAC* forward (RGB-D)");
//	m.def("backward_rgbd", &dsacstar_rgbd_backward, "DSAC* backward (RGB-D)");
}
