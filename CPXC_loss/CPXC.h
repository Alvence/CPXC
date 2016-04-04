#ifndef CPXC_CPXC_H
#define CPXC_CPXC_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CP.h"

class LocalClassifier{
public:
  Pattern *pattern;
  float weight;
  CvNormalBayesClassifier* NBC;
  float singleClass = -1;//if all samples are in the same class

  LocalClassifier(){
    NBC = new CvNormalBayesClassifier();
  }
  float train(cv::Mat &samples, cv::Mat &labels);
  float predict(cv::Mat samples);
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
  void train(PatternSet* patterns, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>*>* mds, LocalClassifier *baseClassifier);
  float predict(cv::Mat sample, vector<int>* matches);
  float predict(cv::Mat sample);
};
#endif
