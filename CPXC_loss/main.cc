#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>


#include <arff_parser.h>
#include <arff_data.h>
#include <arff_value.h>

#include <DPM.h>

#include "BinDivider.h"
#include "CP.h"
#include "Utils.h"
#include "MLAlg.h"

using namespace cv;
using namespace std;

enum ClassifierAlg{ALG_SVM=0, ALG_NN, ALG_NBC};

char datafile[256];
char tempDir[256];
char testfile[256];
int classIndex=-1;
int min_sup = -1;
float min_sup_ratio = 0.01; //default min sup = 1%
int delta = 6;
float prune_threshold = 0.3;
StoppingCreteria sc = NEVER;
bool equalwidth = false;
bool testProvided = false;
ClassifierAlg alg = ALG_NBC;
int cv_fold = 10;
int num_of_attributes;
int num_of_classes;
string algStr = "nbc";

Mat trainingX; 
Mat trainingY;
vector<vector<float>* >*xs = new vector<vector<float>* >();
vector<float> * ys = new vector<float> ();

void analyze_params(int argc, char ** argv){
  int opt= 0;

  //Specifying the expected options
  static struct option long_options[] = {
      {"data",      required_argument, 0,  'd' },
      {"testfile",  optional_argument, 0,  'b' },
      {"tempDir",   required_argument, 0,  't' },
      {"classIndex",optional_argument, 0,  'c' },
      {"min_sup",   optional_argument, 0,  's' },
      {"delta",     optional_argument, 0,  'l' },
      {"stopc",     optional_argument, 0,  'o' },
      {"fold",     optional_argument, 0,  'f' },
      {"eq",     optional_argument, 0,  'e' },
      {"alg",     optional_argument, 0,  'a' },
      {"sr",     optional_argument, 0,  'r' }, 
      {"p_threshold",     optional_argument, 0,  'h' },
      {0,           0,                 0,  0   }
  };

  int long_index =0;
  while ((opt = getopt_long(argc, argv,"ed:t:c:s:l:o:a:b:r:h:", 
                 long_options, &long_index )) != -1) {
    switch (opt) {
      case 'd' : strcpy(datafile,optarg);
        break;
      case 'b' : strcpy(testfile,optarg);
        testProvided = true;
        break;
      case 't' : strcpy(tempDir,optarg);
        break;
      case 'c' : classIndex = atoi(optarg); 
        break;
      case 's' : min_sup = atoi(optarg);
        break;
      case 'l' : delta = atoi(optarg);
        break;
      case 'r':
                 min_sup_ratio = strtof(optarg,NULL);
                 break;
      case 'h':
                 prune_threshold = strtof(optarg,NULL);
                 break;
      case 'a':
        if (strcmp(optarg,"svm") == 0){
          algStr = "svm";
          alg = ALG_SVM;
        } else if (strcmp(optarg, "nn")==0){
          algStr = "nn";
          alg = ALG_NN;
        } else {
          alg = ALG_NBC;
        }
        break;
      case 'o' : 
        if (strcmp(optarg,"threshold") == 0){
          sc = THRESHOLD;
        } else if (strcmp(optarg, "RANDOM")==0){
          sc = RANDOM;
        } else {
          sc = NEVER;
        }
        break;
      case 'f':
        cv_fold = atoi(optarg);
        break;
      case 'e' : equalwidth = true;
        break;
      default: 
               exit(EXIT_FAILURE);
    }
  }
}


template <class T>
void vectorToMat(vector<T>* vec, Mat &mat){
  //TODO makesure dimensions are equal
  for (int i = 0; i < vec->size(); i++){
    mat.at<float>(i,0) = vec->at(i);
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

int get_nominal_index(ArffNominal nominals, String name){
  for (int i = 0; i < nominals.size(); i++){
    if (nominals[i] == name){
      return i;
    }
  }
  return -1;
}

void translate_input(ArffData *ds, Mat &trainingX, Mat& trainingY){
  int cols = 0;
  int num_of_attr = ds->num_attributes();
  int num_of_instances = ds-> num_instances();
  ys->clear();
  xs->clear();
  for (int i = 0; i != num_of_instances; i++){
    cols = 0;
    ArffInstance *ins = ds->get_instance(i);
    vector<float> *x = new vector<float>();
    for (int attr = 0; attr < num_of_attr -1; attr++){
      ArffValue *val = ins->get(attr);
      if (ds->get_attr(attr)->type() == INTEGER||
      ds->get_attr(attr)->type() ==FLOAT||
      ds->get_attr(attr)->type() ==NUMERIC){
        cols++;
        if (val->missing()){
          //cout << "0 ";
          x->push_back(0);
        }else{
          //cout << (float)(*val) << " ";
          x->push_back((float)(*val));
        }
      } else {
        //nominal
        ArffNominal nominals = ds->get_nominal(ds->get_attr(attr)->name());
        int dim = nominals.size();
        for (int k = 0;k < dim; k++){
          cols++;
          if (val->missing()){
            //cout << "0 ";
            x->push_back(0);
          }else{
            if (nominals[k] == (string)(*val)){
              //cout<< "1 ";
              x->push_back(1);
            } else{
              //cout <<"0 ";
              x->push_back(0);
            }
          }
        }
      }
    }
    //output class
    
    //cout << get_nominal_index(ds->get_nominal(ds->get_attr(num_of_attr-1)->name()),(string)(*(ins->get((num_of_attr-1))))) <<" ";
    //cout<<endl;
    ys->push_back(get_nominal_index(ds->get_nominal(ds->get_attr(num_of_attr-1)->name()),(string)(*(ins->get((num_of_attr-1))))));
    xs->push_back(x);
  }
  trainingX = Mat::zeros(num_of_instances,cols,CV_32FC1);
  trainingY = Mat::zeros(num_of_instances,1,CV_32FC1);
  vectorToMat(ys,trainingY);
  vectorsToMat(xs,trainingX);
}

void baseline_classfier_NBC(Mat& trainingX, Mat& trainingY){

  // Train the SVM
  CvNormalBayesClassifier NBC;
  NBC.train(trainingX, trainingY, Mat(), Mat());
 

  float err=0;
  for (int i =0; i< trainingX.rows;i++){
    CvMat sample = trainingX.row(i);
    Mat prob = Mat::zeros(1,3,CV_32FC1);
    CvMat probs = prob;
    cout << i<<": "<<NBC.predict(&sample,0,&probs)<<"   true="<<trainingY.at<float>(i,0)<<" probs:";
    for (int j=0;j<3;j++){
      cout <<" "<<probs.data.fl[j];
    }
    cout<<endl;
    if (fabs(NBC.predict(&sample)- trainingY.at<float>(i,0))>1e-7){
    
      err += 1;
    }
  }
  cout<<"trainning err = "<<err/trainingX.rows<<endl;
  /*
  err=0;
  for (int i = 0; i< testingDataMat.rows;i++){
    if (fabs(SVM.predict(testingDataMat.row(i))-testing_Y->at(i))>1e-7){
    
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  return err*1.0/testingDataMat.rows;*/
}

void baseline_classfier(Mat& trainingX, Mat& trainingY){
   // Set up SVM's parameters
  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

  // Train the SVM
  CvSVM SVM;
  SVM.train(trainingX, trainingY, Mat(), Mat(), params);
  
  float err=0;
  for (int i =0; i< trainingX.rows;i++){
    if (fabs(SVM.predict(trainingX.row(i))- trainingY.at<float>(i,0))>1e-7){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  cout<<"trainning err = "<<err/trainingX.rows<<endl;
  /*
  err=0;
  for (int i = 0; i< testingDataMat.rows;i++){
    if (fabs(SVM.predict(testingDataMat.row(i))-testing_Y->at(i))>1e-7){
    
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  //cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  return err*1.0/testingDataMat.rows;*/
}

//#define CPXC_DEBUG
int main(int argc, char** argv){
  analyze_params(argc,argv);
  int num_patterns;
  char tempDataFile[32];
  char tempTestFile[32];
  strcpy(tempDataFile,tempDir);
  strcpy(tempTestFile,tempDir);
  strcat(tempDataFile,"/binningData.txt");
  strcat(tempTestFile,"/testData.txt");

  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *ds = parser.parse();
  translate_input(ds,trainingX,trainingY);
  
  /*
  for (int row = 0; row < trainingX.rows; row++){
    for (int col = 0; col < trainingX.cols; col++){
      cout << trainingX.at<float>(row,col)<<" ";
    }
    cout <<" class = "<< trainingY.at<float>(row,0)<<endl;
  }*/
  
  baseline_classfier_NBC(trainingX,trainingY);
  
  free(xs);
  free(ys);
  return 0;
}

template <class T>
void free(T* p){
  if (p!=NULL){
    delete p;
    p = NULL;
  }
}

