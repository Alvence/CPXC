#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

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


int get_bin_value(ArffData* ds, string nominal, int attr_index){
  string name = ds->get_attr(attr_index)->name();
  vector<string> nominals = ds->get_nominal(name);
  int value = 0;
  while (value < nominals.size() && nominals[value]!=nominal){
    value++;
  }
  return ((attr_index+1)<<ATTR_SHIFT) + value;
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

void get_instances_for_each_pattern(PatternSet *ps, vector<vector<int>* >* const xs, vector<vector<int>*>* &ins){
  vector<Pattern> patterns = ps->get_patterns();
  for (int i = 0;i < xs->size();i++){
    vector<int>* x = xs->at(i);
    for (int j = 0; j < patterns.size();j++){
      if (patterns[j].match(x)){
        if(ins->at(j) == NULL){
          ins->at(j) = new vector<int> ();
        }
        ins->at(j)->push_back(i);
      }
    }
  }
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


void generate_data(char* file, ArffData* ds, BinDivider* divider, int classIndex, vector<vector<int> *> * &Xs, vector<int> * &t, vector<int> * largeErrSet = 0){
  free_p(Xs);
  free_p(t);
  Xs = new vector<vector<int>*>();
  t = new vector<int>();
  ofstream out;
  out.open(file, ios::out);
  for (int i = 0; i != ds->num_instances(); i++){
    if (largeErrSet!=0){
      if ( std::find(largeErrSet->begin(), largeErrSet->end(), i) == largeErrSet->end() )
      {
        continue;
      }
    }
    ArffInstance* x = ds->get_instance(i);
    ArffValue* y = x->get(classIndex);
    int target = bin_value(y,ds,divider,classIndex);
    //remove higher bits representing the attributes indexs
    target = target & ((1<<ATTR_SHIFT)-1);
    out<<target;
    t->push_back(target);
    vector<int> *ins = new vector<int>();
    for (int j = 0; j != ds->num_attributes(); j++){
      if (j==classIndex) continue;
      ArffValue* v = x->get(j);
      //std::string s = *v;
      if (!v->missing()){
        int val = bin_value(v,ds,divider,j);
        out<<" "<<val;
        ins->push_back(val);
      }
    }
    Xs->push_back(ins);
    out << endl;
  }
  out.close();
}


void baseline_classfier_NBC(Mat& trainingX, Mat& trainingY, vector<int> *largeErrSet, vector<int> *smallErrSet, float rho){

  if(largeErrSet == 0){
    largeErrSet = new vector<int>();
  }
  if(smallErrSet == 0){
    smallErrSet = new vector<int>();
  }

  largeErrSet->clear();
  smallErrSet->clear();


  // Train the SVM
  CvNormalBayesClassifier NBC;
  NBC.train(trainingX.rowRange(0,100), trainingY.rowRange(0,100), Mat(), Mat());
 

  float err=0;
  for (int i =0; i< trainingX.rows;i++){
    CvMat sample = trainingX.row(i);
    Mat prob = Mat::zeros(1,3,CV_32FC1);
    CvMat probs = prob;
    
    float response = NBC.predict(&sample,0,&probs);

    if (fabs(NBC.predict(&sample)- trainingY.at<float>(i,0))>1e-7){
      err += 1;
      largeErrSet->push_back(i);
    } else if (probs.data.fl[(int)response] < rho){
      largeErrSet->push_back(i);
    }else{
      smallErrSet->push_back(i);
    }
  }
  cout<<"trainning err = "<<err/trainingX.rows<<endl;
  cout<<"largeErrSet size = "<<largeErrSet->size()<<" out of "<<trainingX.rows<<endl;
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
 
  num_of_attributes = ds->num_attributes();
  if (classIndex == -1){
    classIndex = num_of_attributes - 1;
  }
  if (min_sup<0){
    min_sup = ds->num_instances()* min_sup_ratio;
    if (min_sup<=1){
      min_sup = 1;
    }
#ifdef CPXC_DEBUG
    cout<<"set min_sup="<<min_sup<<endl;
#endif
  }
  BinDivider* divider= new BinDivider();
  if (equalwidth){
    divider->init_equal_width(ds,5);
  }else{
    divider->init_minimal_entropy(ds, classIndex, sc);
  }
  
  vector<int>* largeErrSet = new vector<int>();
  vector<int>* smallErrSet = new vector<int>();


  baseline_classfier_NBC(trainingX,trainingY,largeErrSet,smallErrSet,20);

  vector<int>* targets = new vector<int>();
  vector<vector<int> *>* newXs = new vector<vector<int> *>();

  generate_data(tempDataFile, ds, divider, classIndex,newXs, targets, largeErrSet);


  num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();
  char tempDPMFile[32];
  strcpy(tempDPMFile,tempDir);
  strcat(tempDPMFile,"/result");
  //generating constrast pattern files
#ifdef CPXC_DEBUG
  printf("generating contrast patterns.\n");
#endif
  num_patterns = dpm(tempDataFile,tempDPMFile,num_of_classes,min_sup,delta);

  cout<<"# of patterns = "<<num_patterns<<endl; 


  PatternSet* patternSet = new PatternSet();
  strcat(tempDPMFile,".closed");
  patternSet->read(tempDPMFile);

  patternSet->print();

  vector<vector<int>* >* ins = new vector<vector<int>* >(patternSet->get_size());

  get_instances_for_each_pattern(patternSet, newXs, ins);


  free(smallErrSet);
  free(largeErrSet);
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

