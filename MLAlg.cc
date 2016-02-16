#include "MLAlg.h"

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

template <class T>
void vectorToMat(vector<T>* vec, Mat &mat){
  //TODO makesure dimensions are equal
  for (int i = 0; i < vec->size(); i++){
    mat.at<float>(i,0) = vec->at(i);
  }
}

template <class T>
void vectorToMat_1ofKCoding(vector<T>* vec, Mat &mat, int num_classes){
  //TODO makesure dimensions are equal
  for (int i = 0; i < vec->size(); i++){
    mat.at<float>(i,vec->at(i)) = 1.0;
  }
}
template <class T>
void vectorsToMat(vector<vector<T>*>*vecs, Mat &mat){
  //TODO make sure dimensions are equal 
  for (int i = 0; i < vecs->size(); i++){
    for (int j = 0; j < vecs->at(i)->size(); j++){
      mat.at<float>(i,j) = vecs->at(i)->at(j);
    }
  }
}

float try_SVM(vector<vector<int>*> *training_X, vector<int> *training_Y, vector<vector<int>* >* testing_X, vector<int> * testing_Y){
  // Set up training data
  Mat labelsMat = Mat::zeros(training_Y->size(), 1 , CV_32FC1);
  Mat trainingDataMat = Mat::zeros(training_X->size(), training_X->at(0)->size(), CV_32FC1);
  vectorToMat(training_Y, labelsMat);
  vectorsToMat(training_X, trainingDataMat);
  //set up testing data
  Mat testingDataMat = Mat::zeros(testing_X->size(), testing_X->at(0)->size(), CV_32FC1);
  vectorsToMat(testing_X, testingDataMat);
  
  // Set up SVM's parameters
  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  // Train the SVM
  CvSVM SVM;
  SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
  
  float err=0;
  for (int i =0; i< trainingDataMat.rows;i++){
    if (fabs(SVM.predict(trainingDataMat.row(i))- training_Y->at(i))>1e-7){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"trainning err = "<<err/trainingDataMat.rows<<endl;
  
  err=0;
  for (int i = 0; i< testingDataMat.rows;i++){
    if (fabs(SVM.predict(testingDataMat.row(i))-testing_Y->at(i))>1e-7){
    
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  return err*1.0/testingDataMat.rows;
}
