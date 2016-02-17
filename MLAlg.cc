#include "MLAlg.h"

#include <vector>
#include <iostream>

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
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

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

//return testing error with n-fold cross validation
float try_SVM(std::vector<std::vector<int>*> *training_X, std::vector<int> *training_Y, int fold){
  vector<float> errs;
  float err = 0.0;
  int N = training_X->size();
  //int N = 59;
  float stepsize = N*1.0/fold;
  float start = 0;
  float end = start + stepsize;
  for (int n = 0; n < fold; n++){
    int first = (int)start;
    start = end;
    int last = (int)end;
    end += stepsize;
    if (end > N){
      end = N;
    }
    //cout << "first = "<<first<<"  last = "<<last<<endl;
    vector<vector<int>*> *x = new vector<vector<int>*>();
    vector<int> *y = new vector<int>();
    vector<vector<int>*> *tx = new vector<vector<int>*>();
    vector<int> *ty = new vector<int>();
    
    for (int i = 0; i < N; i++){
      //extract instances for testing
      if (i<last && i >= first){
        tx->push_back(training_X->at(i));
        ty->push_back(training_Y->at(i));
      } else{
        x->push_back(training_X->at(i));
        y->push_back(training_Y->at(i));
      }
    }
    float cv_err = try_SVM(x,y,tx,ty);
    errs.push_back(cv_err);
    err+=cv_err;
    cout<<"fold "<<n<<":  err="<<cv_err<<"   ["<<first<<","<<last<<")"<<endl;
    delete ty;
    delete tx;
    delete y;
    delete x;
  }
  return err/fold;
}
