#ifndef CPXC_CPXC_H
#define CPXC_CPXC_H

#include <set>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include "CP.h"

using namespace cv;
using namespace cv::ml;

class LocalClassifier{
public:
  Pattern *pattern;
  std::set<int> classes;
  int num_classes;
  float weight;
  float AER;
  Ptr<NormalBayesClassifier> NBC;
  float singleClass = -1;//if all samples are in the same class

  LocalClassifier(){
    NBC =NormalBayesClassifier::create();
  }
  float train(cv::Mat &samples, cv::Mat &labels);
  float predict(cv::Mat samples);
  float predict(Mat sample, Mat& probs);
};

class CPXC{
private:
  std::vector<int>* getMatches(cv::Mat sample);

public:
  std::vector<LocalClassifier*> * classifiers;
  LocalClassifier* defaultClassifier;
  int num_of_classes;

  CPXC(){
    classifiers = new std::vector<LocalClassifier*>();
  }
  ~CPXC(){
    delete classifiers;
    classifiers = NULL;
  }
  void train(PatternSet* patterns, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>*>* mds, LocalClassifier *baseClassifier,int); 
  float predict(cv::Mat sample, vector<int>* matches);
  float predict(cv::Mat sample);
};
#endif
