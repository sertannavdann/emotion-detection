#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iterator>
#include <deque>
#include <random>

using namespace std;
using namespace cv;

const vector<string> classes = { "fear", "angry", "sad", "neutral", "surprise", "disgust", "happy" };

constexpr int64_t numChannels = 3;
constexpr int64_t imageHeight = 128;
constexpr int64_t imageWidth = 128;
constexpr int64_t numClasses = 7;
constexpr int64_t numInputElements = numChannels * imageHeight * imageWidth;

const auto model_path = L"C:\\Users\\Serta\\Desktop\\CPP\\FaceDetection\\FaceDetectionConsole\\assets\\FER_Model_Adam.onnx";
const string input_name = "input.1";
const string output_name = "255";

const string window_name = "Face Detection";

constexpr int maxHistory = 50;

vector<Scalar> randomColors(int numColors);
vector<float> loadImage(cv::Mat image, int sizeX, int sizeY);
vector<float> face2Int(cv::Mat face);
void drawDetectedFeatures(cv::Mat& image, const vector<cv::Rect>& features);
void captureVideoAndProcess(cv::CascadeClassifier& classifier);
void visualizeProbabilities(const vector<float>& probabilities, const vector<string>& classes, Mat& graph, deque<vector<float>>& history, int maxHistory, const vector<Scalar>& randomColorsVec);