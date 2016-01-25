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

using namespace cv;
using namespace std;

char datafile[256];
char tempDir[256];
int classIndex;
int min_sup = 1;
int delta = 100;

int num_of_attributes;
int num_of_classes;

vector<int> targets;
vector<vector<int> > xs;
vector<vector<int> > binning_xs;

int get_bin_value(ArffData* ds, string nominal, int attr_index){
  string name = ds->get_attr(attr_index)->name();
  vector<string> nominals = ds->get_nominal(name);
  int value = 0;
  while (value < nominals.size() && nominals[value]!=nominal){
    value++;
  }
  return (attr_index+1)*10 + value;
}

int bin_value(ArffValue *v, ArffData* ds, BinDivider* divider, int index){
  int str=-1;
  if (v->type() == FLOAT || v->type() == INTEGER){
    str = divider->get_bin_value((float)(*v),index);
  }else if(v->type()==STRING||v->type()==DATE){
    str = get_bin_value(ds,(string)(*v),index);
  }
  return str;
}

void generate_binning_data(char* file, ArffData* ds, BinDivider* divider, int classIndex){
  ofstream out;
  out.open(file, ios::out);
  for (int i = 0; i != ds->num_instances(); i++){
    ArffInstance* x = ds->get_instance(i);
    ArffValue* y = x->get(classIndex);
    int target = bin_value(y,ds,divider,classIndex)%10;
    out<<target;
    targets.push_back(target);
    vector<int> ins;
    for (int j = 0; j != ds->num_attributes(); j++){
      if (j==classIndex) continue;
      ArffValue* v = x->get(j);
      //std::string s = *v;
      if (!v->missing()){
        int val = bin_value(v,ds,divider,j);
        out<<" "<<val;
        ins.push_back(val);
      }
    }
    xs.push_back(ins);
    out << endl;
  }
  out.close();
}

void try_SVM(){
  // Set up training data
  Mat labelsMat(targets.size(), 1, CV_32FC1);
  for (int i=0; i<targets.size();i++){
    labelsMat.at<float>(i,0) = targets[i];
  }

  Mat trainingDataMat(binning_xs.size(), binning_xs[0].size(), CV_32FC1);
  for (int i=0; i<binning_xs.size();i++){
    for (int j=0; j<binning_xs[i].size();j++){
      trainingDataMat.at<float>(i,j) = binning_xs[i][j];
    }
  }


  // Set up SVM's parameters
  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  // Train the SVM
  CvSVM SVM;
  SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
  float err=0;
  for (int i =0; i<binning_xs.size();i++){
    Mat sample = trainingDataMat.row(i);
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
    if (SVM.predict(sample)!=targets[i]){
      err += 1;
    }
  }
  cout<<"err: "<<err/targets.size()<<endl;
}


void try_NN(){
  //70% data for training
  int breakpoint = (targets.size()*7)/10;
  // Set up training data
  Mat labelsMat(breakpoint, num_of_classes, CV_32FC1);
  for (int i=0; i<breakpoint;i++){
    labelsMat.at<float>(i,targets[i]) = 1.0;
  }

  Mat trainingDataMat(breakpoint, binning_xs[0].size(), CV_32FC1);
  for (int i=0; i<breakpoint;i++){
    for (int j=0; j<binning_xs[i].size();j++){
      trainingDataMat.at<float>(i,j) = binning_xs[i][j];
    }
  }

  
  Mat testLabelsMat(targets.size()-breakpoint, num_of_classes, CV_32FC1);
  for (int i=breakpoint; i<targets.size();i++){
    testLabelsMat.at<float>(i,targets[i]) = 1.0;
  }

  Mat testingDataMat(binning_xs.size()-breakpoint, binning_xs[0].size(), CV_32FC1);
  for (int i=breakpoint; i<binning_xs.size();i++){
    for (int j=0; j<binning_xs[i].size();j++){
      testingDataMat.at<float>(i,j) = binning_xs[i][j];
    }
  }

  cout<<binning_xs[0].size()<<" "<<num_of_classes<<endl;
  int layers_d[] = { binning_xs[0].size(),binning_xs[0].size()/2,10, 10,  num_of_classes};
  Mat layers = Mat(1,5,CV_32SC1);
  for (int i = 0; i < 5; i++){
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
  cout<<"finish training with "<<iterations<<" iterations"<<endl;
  
  float err=0;
  for (int i =0; i<breakpoint;i++){
    Mat response;
    nnetwork->predict(trainingDataMat.row(i),response);
    int res = -1;
    float max = 0;
    for (int j = 0;j<num_of_classes;j++){
      if (response.at<float>(0,j) > max){
        res = j;
      }
    }
    if (res!=targets[i]){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  cout<<"trainning err = "<<err/breakpoint<<endl;
  
  err=0;
  for (int i = 0; i< testingDataMat.rows;i++){
    Mat response;
    nnetwork->predict(testingDataMat.row(i),response);
    int res = -1;
    float max = 0;
    for (int j = 0;j<num_of_classes;j++){
      if (response.at<float>(0,j) > max){
        res = j;
      }
    }
    if (res!=targets[i+breakpoint]){
    //cout << SVM.predict(sample)<<"   true="<<targets[i]<<endl;
      err += 1;
    }
  }
  cout<<"testing err = "<<err/testingDataMat.rows<<endl;
  
  delete nnetwork;
}

void translate_input(PatternSet* ps){
  binning_xs.clear();
  for (vector<vector<int> >::iterator it = xs.begin(); it!=xs.end(); it++){
    binning_xs.push_back(ps->translate_input(*it));
  }
}

void print_usage() {
  printf("Usage: \n");
}
  


void analyze_params(int argc, char ** argv){
  int opt= 0;

  //Specifying the expected options
  //The two options l and b expect numbers as argument
  static struct option long_options[] = {
      {"data",      required_argument, 0,  'd' },
      {"tempDir",   required_argument, 0,  't' },
      {"classIndex",required_argument, 0,  'c' },
      {"min_sup",   required_argument, 0,  's' },
      {"delta",     required_argument, 0,  'l' },
      {0,           0,                 0,  0   }
  };

  int long_index =0;
  while ((opt = getopt_long(argc, argv,"d:t:c:s:l:", 
                 long_options, &long_index )) != -1) {
    switch (opt) {
      case 'd' : strcpy(datafile,optarg);
        break;
      case 't' : strcpy(tempDir,optarg);
        break;
      case 'c' : classIndex = atoi(optarg); 
        break;
      case 's' : min_sup = atoi(optarg);
        break;
      case 'l' : delta = atoi(optarg);
        break;
      default: print_usage(); 
               exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char** argv){
  analyze_params(argc,argv);

  char tempDataFile[32];
  strcpy(tempDataFile,tempDir);
  strcat(tempDataFile,"/binningData.txt");

  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *ds = parser.parse();
  
  printf("Successfully finish reading the data!\n");

  num_of_attributes = ds->num_attributes();
  BinDivider* divider= new BinDivider();
  divider->init(ds,5);

  //sava binning temp data file
  printf("saving binning data to %s\n",tempDataFile);
  generate_binning_data(tempDataFile, ds, divider, classIndex);

  num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();
  
  char tempDPMFile[32];
  strcpy(tempDPMFile,tempDir);
  strcat(tempDPMFile,"/result");
  //generating constrast pattern files
  printf("generating contrast patterns.\n");
  dpm(tempDataFile,tempDPMFile,num_of_classes,min_sup,delta);

  //read patterns
  PatternSet* patternSet = new PatternSet();
  strcat(tempDPMFile,".closed");
  patternSet->read(tempDPMFile);
  //patternSet->print();

  //translate input
  cout<<"translating input"<<endl;
  translate_input(patternSet);
  
  /*for (int i = 0;i<xs.size();i++){
    print_vector(xs[i]);
    print_vector(binning_xs[i]);
  }*/
 
  //try neural network
  try_NN();

  free(patternSet);
  free(divider);
  return 0;
}

template <class T>
void free(T* p){
  if (p!=NULL){
    delete p;
    p = NULL;
  }
}

