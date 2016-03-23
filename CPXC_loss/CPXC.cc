#include "CPXC.h"

#include <cmath>
using namespace cv;
using namespace std;
float LocalClassifier::train(Mat & trainingX, Mat & trainingY){
  NBC->train(trainingX, trainingY, Mat(), Mat());
}

float LocalClassifier::predict(Mat sample){
  if (sample.rows>1){
    return 0;
  }
  if (singleClass >= 0){
    return singleClass;
  }
  return NBC->predict(sample);
}


void CPXC::train(PatternSet* patterns, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>*>* mds, LocalClassifier *baseClassifier){
  int num_ins = xs.rows;
  vector<int> flags(num_ins,0);
  classifiers->clear();
  for (int i = 0; i < mds->size(); i++){
    vector<int>* md = mds->at(i);
    if (md->size()<1){
      continue;
    }
    LocalClassifier* cf = new LocalClassifier();
    cf->weight = md->size();
    Mat trainingX = Mat::zeros(md->size(),xs.cols, CV_32FC1);
    Mat trainingY = Mat::zeros(md->size(),1,CV_32FC1);
    for (int j = 0; j < md->size(); j++){
      int index = md->at(j);
      for (int col = 0; col < trainingX.cols; col++){
        trainingX.at<float>(j,col)=xs.at<float>(index,col);
      }
      trainingY.at<float>(j,0)=ys.at<float>(index,0);
      //cout<<"("<<index<<","<<j<<")"<<endl;
      //xs.row(index).copyTo(trainingX.row(j));
      //ys.row(index).copyTo(trainingY.row(j));
      flags[index] = 1;
    }
    ////for (int k=0;k < md->size();k++){
    ////  cout<< md->at(k)<<":"<<trainingY.at<float>(k,0)<<":"<<ys.at<float>(md->at(k),0)<<"  ";
    ////}cout<<endl;

    //check whether it is single class
    bool flag = true;
    float label = trainingY.at<float>(0,0);
    for (int k = 1; k < trainingY.rows;k++){
      if (fabs(trainingY.at<float>(k,0)-label)>1e-5){
        flag = false;
        break;
      }
    }
    if (!flag){
      cf->pattern = &((patterns->get_patterns())[i]);
      cf->train(trainingX,trainingY);
      //cout<<"complete training for pattern" << i<<endl;
    }else{
      cf->singleClass = label;
    }
    classifiers->push_back(cf);
  }
  Mat dtrainingX;
  Mat dtrainingY;
  int j=0;
  for (int i = 0; i < num_ins;i++){
    if(flags[i]!=1){
      dtrainingX.push_back(xs.row(i));
      dtrainingY.push_back(ys.row(i));
    }
  }
  if (dtrainingX.rows>0){
    defaultClassifier = new LocalClassifier();
    defaultClassifier->train(dtrainingX,dtrainingY);
  }
  else{
    defaultClassifier = baseClassifier;
  }
}

float CPXC::predict(cv::Mat sample, vector<int>* matches){
  if (matches->size() == 0){
    return defaultClassifier->predict(sample);
  }
  vector<float> votes(num_of_classes,0);
  for (int i =0; i < matches->size(); i++){
    float response = classifiers->at(matches->at(i))->predict(sample);
    votes[(int)response] += classifiers->at(matches->at(i))->weight;
  }
  int label;
  float maxVote=-1;
  for (int i = 0; i < num_of_classes;i++){
    if (votes[i]>maxVote){
      label = i;
      maxVote = votes[i];
    }
  }
  return label;
}
