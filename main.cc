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
StoppingCreteria sc = NEVER;
bool equalwidth = false;
bool testProvided = false;
ClassifierAlg alg = ALG_NBC;
int cv_fold = 10;
int num_of_attributes;
int num_of_classes;
string algStr = "nbc";

vector<int>* targets=NULL;
vector<vector<int>*> *xs=NULL;
vector<vector<int>*> *binning_xs=NULL;
vector<int> * test_targets=NULL;
vector<vector<int>*> *test_binning_xs=NULL;
vector<vector<int>*> *test_xs=NULL;

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

void generate_binning_data(char* file, ArffData* ds, BinDivider* divider, int classIndex, vector<vector<int> *> * &Xs, vector<int> * &t){
  free_p(Xs);
  free_p(t);
  Xs = new vector<vector<int>*>();
  t = new vector<int>();
  ofstream out;
  out.open(file, ios::out);
  for (int i = 0; i != ds->num_instances(); i++){
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

void translate_input(PatternSet* ps, vector<vector<int>*>* const binning, vector<vector<int>*>* &res){
  if(res == NULL){
    res = new vector<vector<int>*>();
  }else{
    res->clear();
  }
  for (vector<vector<int> *>::iterator it = binning->begin(); it!=binning->end(); it++){
    res->push_back(new vector<int>(ps->translate_input(*(*it))));
  }
}

void print_usage() {
  printf("Usage: \n");
}
  

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
      {0,           0,                 0,  0   }
  };

  int long_index =0;
  while ((opt = getopt_long(argc, argv,"ed:t:c:s:l:o:a:b:r:", 
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
      default: print_usage(); 
               exit(EXIT_FAILURE);
    }
  }
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

  ArffParser *test_parser = NULL;
  ArffData *test_ds = NULL;
  //open test data
  if (testProvided){
    test_parser = new ArffParser(testfile);
    printf("parse test file!!!!!!!\n");
    test_ds= test_parser->parse();
  }
#ifdef CPXC_DEBUG
  printf("Successfully finish reading the data!\n");
#endif
  ///cout<<"num of instances="<<ds->num_instances()<<endl;

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
  //sava training temp data file
#ifdef CPXC_DEBUG
  printf("saving training data to %s\n",tempDataFile);
#endif
  targets = new vector<int>();
  generate_binning_data(tempDataFile, ds, divider, classIndex,binning_xs, targets);
  

  if (testProvided){
    //save testing temp data file
#ifdef CPXC_DEBUG
    printf("saving testing data to %s\n", tempTestFile);
#endif
    generate_binning_data(tempTestFile, test_ds, divider, classIndex, test_binning_xs, test_targets);
  }
  num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();
  
  char tempDPMFile[32];
  strcpy(tempDPMFile,tempDir);
  strcat(tempDPMFile,"/result");
  //generating constrast pattern files
#ifdef CPXC_DEBUG
  printf("generating contrast patterns.\n");
#endif
  num_patterns = dpm(tempDataFile,tempDPMFile,num_of_classes,min_sup,delta);

  //read patterns
  PatternSet* patternSet = new PatternSet();
  strcat(tempDPMFile,".closed");
  patternSet->read(tempDPMFile);

  //patternSet->prune_AMI(binning_xs);
  //patternSet->print();

  //translate input
  translate_input(patternSet, binning_xs, xs);

  if (testProvided){
    translate_input(patternSet, test_binning_xs,test_xs);
  }
  //for (int i = 0;i<xs.size();i++){
  //  print_vector(xs[i]);
  //  print_vector(binning_xs[i]);
  //}
#ifdef CPXC_DEBUG
  cout<<"start training and testing"<<endl;
#endif
  float err = 0.0;
  switch(alg){
    case ALG_SVM:
      if (testProvided){
        err=try_SVM(xs,targets,test_xs,test_targets);
      } else{
        err=try_SVM(xs,targets,cv_fold);
      }
      break;
    case ALG_NN:
      if (testProvided){
        err=try_NN(xs,targets,test_xs,test_targets, num_of_classes);
      } else{
        err=try_NN(xs,targets, num_of_classes, cv_fold);
      }
      break;
    case ALG_NBC:
      if (testProvided){
        err=try_NBC(xs,targets,test_xs,test_targets);
      } else{
        err=try_NBC(xs,targets,cv_fold);
      }
      break;
    default:
      break;

  }
  printf("%s: alg = %s  pattern=%d min_sup=(%.0f%,%d) ratio=%d  error = %.2f %\n",datafile,algStr.c_str(), num_patterns, min_sup_ratio*100,min_sup, delta,err*100);
  //try neural network
  //try_NN();
  //try_SVM();
  //try_NBC();
  free(test_parser);
  free(binning_xs);
  free(xs);
  free(targets);
  free(test_binning_xs);
  free(test_xs);
  free(test_targets);
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

