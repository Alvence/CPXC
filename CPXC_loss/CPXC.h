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
  Pattern pattern;
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

  void statBinaryCase(cv::Mat& samples, cv::Mat &labels, float & acc, float& fscore, int & num_c1, int &num_c2);
};

class CPXC{

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
  float predict(cv::Mat sample, vector<int>* matches, cv::Mat& probs);
  float predict(cv::Mat sample, vector<int>* matches);
  float predict1(cv::Mat sample, vector<int>* bin_ins);
  float predict1(cv::Mat sample, vector<int>* bin_ins, cv::Mat &probs);
  void save(char* filename);
  void print_cover(vector<int> statCov);
  std::vector<int>* getMatches(vector<int>* ins);
  float TER(Ptr<NormalBayesClassifier> base, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>* >* bin_xs);
  float obj(Ptr<NormalBayesClassifier> base, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>* >* bin_xs);
  CPXC optimize(int k,Ptr<NormalBayesClassifier> base, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>* >* bin_xs);
  void print_pattern_cover(vector<vector<int>* >* xs);
  void printPatternStat(Mat& samples, Mat& labels, vector<vector<int>* >* bin_xs);
  void printPatternLS(vector<vector<int>* >* bin_xs,vector<int>* LE);
};
#endif
