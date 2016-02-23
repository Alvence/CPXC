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
#ifdef CPXC_DEBUG
    cout<<"fold "<<n<<":  err="<<cv_err<<"   ["<<first<<","<<last<<")"<<endl;
#endif
    delete ty;
    delete tx;
    delete y;
    delete x;
  }
  return err/fold;
}

float try_NBC(vector<vector<int>*> *training_X, vector<int> *training_Y, vector<vector<int>* >* testing_X, vector<int> * testing_Y){
  // Set up training data
  Mat labelsMat = Mat::zeros(training_Y->size(), 1 , CV_32FC1);
  Mat trainingDataMat = Mat::zeros(training_X->size(), training_X->at(0)->size(), CV_32FC1);
  vectorToMat(training_Y, labelsMat);
  vectorsToMat(training_X, trainingDataMat);
  //set up testing data
  Mat testingDataMat = Mat::zeros(testing_X->size(), testing_X->at(0)->size(), CV_32FC1);
  vectorsToMat(testing_X, testingDataMat);

  // Train the SVM
  CvNormalBayesClassifier NBC;
  NBC.train(trainingDataMat, labelsMat, Mat(), Mat());
  
  float err=0;
  /*for (int i =0; i<trainingDataMat.rows;i++){
    if (fabs(NBC.predict(trainingDataMat.row(i))- training_Y->at(i))>1e-7){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  ///cout<<"trainning err = "<<err/trainingDataMat.rows<<endl;
  
  err=0;*/
  for (int i = 0; i< testingDataMat.rows;i++){
    if (fabs(NBC.predict(testingDataMat.row(i))-testing_Y->at(i))>1e-7){
    
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  return err/testingDataMat.rows;
}


//return testing error with n-fold cross validation
float try_NBC(std::vector<std::vector<int>*> *training_X, std::vector<int> *training_Y, int fold){
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
    float cv_err = try_NBC(x,y,tx,ty);
    errs.push_back(cv_err);
    err+=cv_err;
#ifdef CPXC_DEBUG
    cout<<"fold "<<n<<":  err="<<cv_err<<"   ["<<first<<","<<last<<")"<<endl;
#endif
    delete ty;
    delete tx;
    delete y;
    delete x;
  }
  return err/fold;
}

float try_NN(vector<vector<int>*> *training_X, vector<int> *training_Y, vector<vector<int>* >* testing_X, vector<int> * testing_Y, int num_of_classes){
  // Set up training data
  Mat labelsMat = Mat::zeros(training_Y->size(), num_of_classes , CV_32FC1);
  Mat trainingDataMat = Mat::zeros(training_X->size(), training_X->at(0)->size(), CV_32FC1);
  vectorToMat_1ofKCoding(training_Y, labelsMat, num_of_classes);
  vectorsToMat(training_X, trainingDataMat);
  //set up testing data
  Mat testingDataMat = Mat::zeros(testing_X->size(), testing_X->at(0)->size(), CV_32FC1);
  vectorsToMat(testing_X, testingDataMat);
#ifdef CPXC_DEBUG 
  cout<<training_X->at(0)->size()<<" "<<num_of_classes<<endl;
#endif
  int layers_d[] = { training_X->at(0)->size(), 20,  num_of_classes};
  Mat layers = Mat(1,3,CV_32SC1);
  for (int i = 0; i < 3; i++){
#ifdef CPXC_DEBUG
    cout << "layer "<<i<<" :"<<layers_d[i]<<endl;
#endif
    layers.at<int>(0,i) = layers_d[i];
  }
  // create the network using a sigmoid function with alpha and beta
  //parameters 0.6 and 1 specified respectively (refer to manual)

  CvANN_MLP* nnetwork = new CvANN_MLP;
  nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

  // set the training parameters

  // terminate the training after either 1000
  // iterations or a very small change in the
  // network wieghts below the specified value
 
  CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(
    cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
    CvANN_MLP_TrainParams::BACKPROP,
    0.1,
    0.1
  );
  int iterations = nnetwork->train(trainingDataMat, labelsMat, Mat(), Mat(), params);
  ///cout<<"finish training with "<<iterations<<" iterations"<<endl;
  
  float err=0;
  /*for (int i =0; i<trainingDataMat.rows;i++){
    Mat response;
    nnetwork->predict(trainingDataMat.row(i),response);
    int res = -1;
    float max = 0;
    for (int j = 0;j<num_of_classes;j++){
      if (response.at<float>(0,j) > max){
        res = j;
        max = response.at<float>(0,j);
      }
    }
    if (res!=training_Y->at(i)){
      err += 1;
    }
  }
  cout<<"trainning err = "<<err/trainingDataMat.rows<<endl;
  
  err=0;*/
  for (int i = 0; i< testingDataMat.rows;i++){
    Mat response;
    nnetwork->predict(testingDataMat.row(i),response);
    int res = -1;
    float max = 0;
    for (int j = 0;j<num_of_classes;j++){
      if (response.at<float>(0,j) > max){
        res = j;
        max = response.at<float>(0,j);
      }
    }
    if (res!= testing_Y->at(i)){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  
  delete nnetwork;
  return err/testingDataMat.rows;
}
//return testing error with n-fold cross validation
float try_NN(std::vector<std::vector<int>*> *training_X, std::vector<int> *training_Y, int num_of_classes, int fold){
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
    float cv_err = try_NN(x,y,tx,ty, num_of_classes);
    errs.push_back(cv_err);
    err+=cv_err;
#ifdef CPXC_DEBUG
    cout<<"fold "<<n<<":  err="<<cv_err<<"   ["<<first<<","<<last<<")"<<endl;
#endif
    delete ty;
    delete tx;
    delete y;
    delete x;
  }
  return err/fold;
}
