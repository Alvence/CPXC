#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <math.h>
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
#include <gcgrowth.h>

#include "BinDivider.h"
#include "CP.h"
#include "Utils.h"
//#include "MLAlg.h"
#include "CPXC.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

enum ClassifierAlg{ALG_SVM=0, ALG_NN, ALG_NBC};

char datafile[256];
char tempDir[256];
char testfile[256];
int classIndex=-1;
int min_sup = -1;
float min_sup_ratio = 0.02; //default min sup = 2%
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
float ro = 0.45;

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
      {"rho",     optional_argument, 0,  'i' },
      {"eq",     optional_argument, 0,  'e' },
      {"alg",     optional_argument, 0,  'a' },
      {"sr",     optional_argument, 0,  'r' }, 
      {"p_threshold",     optional_argument, 0,  'h' },
      {0,           0,                 0,  0   }
  };

  int long_index =0;
  while ((opt = getopt_long(argc, argv,"ed:t:c:s:l:o:a:b:r:h:f:i:", 
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
      case 'i':  ro = strtof(optarg,NULL);
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

void get_instances_for_each_pattern(PatternSet *ps, vector<vector<int>* >* const xs, vector<vector<int>*>* &ins, vector<int>* largeErrSet){
  vector<Pattern> patterns = ps->get_patterns();
  for (int i = 0;i < xs->size();i++){
    vector<int>* x = xs->at(i);
    for (int j = 0; j < patterns.size();j++){
      if (patterns[j].match(x)){
        if(ins->at(j) == NULL){
          ins->at(j) = new vector<int> ();
        }
        ins->at(j)->push_back(largeErrSet->at(i));
      }
    }
  }
}


void translate_input(ArffData *ds, Mat &trainingX, Mat& trainingY){
  int cols = 0;
  vector<vector<float>* >*xs = new vector<vector<float>* >();
  vector<float> * ys = new vector<float> ();
  int num_of_attr = ds->num_attributes();
  int num_of_instances = ds-> num_instances();
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
  trainingY = Mat::zeros(num_of_instances,1,CV_32SC1);
  vectorToMat(ys,trainingY);
  vectorsToMat(xs,trainingX);
  free(xs);
  free(ys);
}

vector<int>* get_matches(ArffData *ds, BinDivider* divider, PatternSet* ps,int classIndex, int index){
  vector<int>* res = new vector<int>();
  vector<int>* ins = new vector<int>();
  
  ArffInstance* x = ds->get_instance(index);

  for (int j = 0; j != ds->num_attributes(); j++){
    if (j==classIndex) continue;
    ArffValue* v = x->get(j);
    //std::string s = *v;
    if (!v->missing()){
      int val = bin_value(v,ds,divider,j);
      ins->push_back(val);
    }
  }

  for (int i = 0; i < ps->get_patterns().size();i++){
    if (ps->get_patterns().at(i).match(ins)){
      res->push_back(i);
    }
  }

  free(ins);
  return res;
}
void generate_data(char* file, ArffData* ds, BinDivider* divider, int classIndex, vector<vector<int> *> * &Xs,  vector<int> * &t, vector<vector< vector<int> *> *>* & labelledXs, vector<int> * largeErrSet){
  free_p(Xs);
  free_p(t);
  int num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();
  labelledXs = new vector<vector<vector<int> *> *>(num_of_classes);
  for (int c=0;c<num_of_classes;c++){
    labelledXs->at(c) = new vector<vector<int>* >();
  }
  Xs = new vector<vector<int>*>();
  t = new vector<int>();
  ofstream out;
  out.open(file, ios::out);
  for (int k = 0; k != largeErrSet->size(); k++){
    int index = largeErrSet->at(k);
    ArffInstance* x = ds->get_instance(index);
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
    labelledXs->at(target)->push_back(ins);
    Xs->push_back(ins);
    out << endl;
  }
  out.close();
}

void generate_data(char* file, ArffData* ds, BinDivider* divider, int classIndex, vector<vector<int> *> * &Xs,vector<int> * &t){ 
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


Ptr<NormalBayesClassifier> baseline_classfier_NBC(Mat& trainingX, Mat& trainingY, vector<int> *largeErrSet, vector<int> *smallErrSet,int num_classes,  float rho){

  if(largeErrSet == 0){
    largeErrSet = new vector<int>();
  }
  if(smallErrSet == 0){
    smallErrSet = new vector<int>();
  }

  largeErrSet->clear();
  smallErrSet->clear();


  // Train the SVM
  Ptr<NormalBayesClassifier> NBC = NormalBayesClassifier::create();
  NBC->train(trainingX, ROW_SAMPLE, trainingY);
  float err=0;
  int N = trainingX.rows;
  int size = 0;
  float r = 0;
  float low = 10000;
  float high = -10000;

  for (float ratio = -100; ratio < 0; ){
    int tempSize = 0;
    for (int i =0; i< trainingX.rows;i++){
      Mat sample = trainingX.row(i);
      Mat probs;
      Mat result;
      
      NBC->predictProb(sample,result,probs);
  
      float V = 0;
      for (int c = 0; c < num_classes; c++){
        V+=probs.at<float>(0,c); 
      }
      int res = (int) result.at<float>(0,0);
      if (fabs(res - trainingY.at<float>(i,0))>1e-7){
        tempSize++;
      } else if (log10(probs.at<float>(0,res)/V)*100 < ratio){
        tempSize++;
      }
    }
    if (fabs(tempSize - N*rho) < fabs(size - N*rho)){
      size = tempSize;
      r = ratio;
    }
    if(ratio < -2){
      ratio +=1;
    }else{
      ratio += 0.001;
    }
  }

  //cout<<"si9ze=" <<size<<endl;
  for (int i =0; i< trainingX.rows;i++){
    Mat sample = trainingX.row(i);
    Mat probs = Mat::zeros(num_classes,1,CV_32FC1);
    Mat result;

    int response; 
    NBC->predictProb(sample,result,probs);
    response = (int) result.at<float>(0,0);

    //cout<<"instance "<<i<<" resp="<<result.at<float>(0,0)<<" r="<<r;
    float V = 0;
    for (int c = 0; c < num_classes; c++){
      V+=probs.at<float>(0,c); 
    }
    /*
    for (int c = 0; c < num_classes; c++){
      cout<<" "<<(probs.at<float>(0,c)); 
    }
    cout<<endl;*/
    float prob = log10(probs.at<float>(0,response)/V)*100;

    if (prob < low ){
      low =  prob; 
    }
    if (prob>high ){
      high =  prob; 
    }
    if ((fabs(response)- trainingY.at<float>(i,0))>1e-7){
      err += 1;
      largeErrSet->push_back(i);
    } else if (prob < r){
      largeErrSet->push_back(i);
    }else{
      smallErrSet->push_back(i);
    }
  }
  //cout<<"trainning err = "<<err/trainingX.rows<<endl;
  cout<<"largeErrSet size = "<<largeErrSet->size()<<" out of "<<trainingX.rows <<"high = "<<high<<"  low ="<<low<<endl;
  return NBC;
}


//#define CPXC_DEBUG
float run(int argc, char** argv, int first, int last){
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
  ArffData *dsa = parser.parse();
 
  ArffData *ds = new ArffData();
  ArffData *testingds = new ArffData();

  dsa->split(testingds, ds, first, last);
  num_of_attributes = ds->num_attributes();
  //cout<<trainingds->num_instances()<<"   "<<testingds->num_instances()<<endl;
  if (classIndex == -1){
    classIndex = num_of_attributes - 1;
  }
  num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();

  Mat trainingX; 
  Mat trainingY;
  Mat testingX;
  Mat testingY;

  translate_input(ds, trainingX,trainingY);
  translate_input(testingds, testingX,testingY);

  BinDivider* divider= new BinDivider();
  if (equalwidth){
    divider->init_equal_width(ds,5);
  }else{
    divider->init_minimal_entropy(ds, classIndex, sc);
  }
  
  vector<int>* largeErrSet = new vector<int>();
  vector<int>* smallErrSet = new vector<int>();

  Ptr<NormalBayesClassifier> nbc= baseline_classfier_NBC(trainingX,trainingY,largeErrSet,smallErrSet,num_of_classes,ro);
  
  if (min_sup<0){
    min_sup = largeErrSet->size()* min_sup_ratio;
    if (min_sup<=1){
      min_sup = 1;
    }
#ifdef CPXC_DEBUG
    cout<<"set min_sup="<<min_sup<<endl;
#endif
  }
  vector<int>* targets = new vector<int>();
  vector<vector<int> *>* newXs = new vector<vector<int> *>();
  vector< vector<vector<int> *> *>* labelledXs;

  generate_data(tempDataFile, ds, divider, classIndex,newXs, targets, labelledXs,largeErrSet);
  char tempDPMFile[32];
  strcpy(tempDPMFile,tempDir);
  strcat(tempDPMFile,"/result");
  //generating constrast pattern files
#ifdef CPXC_DEBUG
  printf("generating contrast patterns.\n");
#endif
  vector<int> classes(num_of_classes,0);
  for (int i = 0;i < trainingY.rows; i++){
    classes[(int)trainingY.at<float>(i,0)] = 1;
  }
  int real_num_class = 0;
  for (int i = 0; i < classes.size(); i++){
    real_num_class += classes[i];
  }

  //num_patterns = dpm(tempDataFile,tempDPMFile,real_num_class,min_sup,delta);
  num_patterns = gcgrowth(tempDataFile,tempDPMFile,min_sup);

  PatternSet* patternSet = new PatternSet();
  strcat(tempDPMFile,".key");
  patternSet->read(tempDPMFile);

  //patternSet->print();

  cout<<"pattern number="<<patternSet->get_size()<<endl;

  patternSet->filter(newXs,labelledXs,num_of_attributes,delta);


  cout<<"pattern number="<<patternSet->get_size()<<endl;


  vector<vector<int>* >* ins = new vector<vector<int>* >(patternSet->get_size());

  get_instances_for_each_pattern(patternSet, newXs, ins, largeErrSet);

  CPXC classifier;
  classifier.num_of_classes = num_of_classes;
  LocalClassifier* base = new LocalClassifier();
  base->NBC = nbc;
  //cout<<patternSet<<"  "<<trainingX.rows<<" "<<trainingY.rows<<" "<<ins->size()<<" "<<base<<endl;
  classifier.train(patternSet,trainingX,trainingY,ins,base,num_of_classes);
  int err =0;
  cout<<"classifier number = "<< classifier.classifiers->size()<<endl;
      /*float errReduction=0.0;
      for (int j = 0; j < trainingX.rows; j++){
        Mat probs1;
        Mat probs2;
        Mat result1;
        Mat result2;
        
        cf->predictProb(trainingX.row(j),result1,probs1);
        baseClassifier->predictProb(trainingX.row(j),result2,probs2);

        int res1 = (int) result1.at<float>(0,0);
        int res2 = (int) result2.at<float>(0,0);
        float norm1 = 0.0;
        float norm2 = 0.0;
        for (int c = 0; c < probs.cols;c++){
          norm1+=probs1.at<float>(0,c);
          norm2+=probs2.at<float>(0,c);
        }
        float err1 = fabs(1 - probs1.at<float>(0,(int)trainingY.at<float>(j,0))/norm1);
        float err2 = fabs(1 - probs2.at<float>(0,(int)trainingY.at<float>(j,0))/norm2);
        errReduction += fabs(err1-err2);
      }
      errReduction/= md->size();*/
  vector<int> TP(num_of_classes,0);
  vector<int> FP(num_of_classes,0);
  vector<int> TN(num_of_classes,0);
  vector<int> FN(num_of_classes,0);
  for (int i = 0; i < testingds->num_instances();i++){
    vector<int>* md = get_matches(testingds,divider,patternSet,classIndex,i);
    int response = (int)classifier.predict(testingX.row(i),md);
    int trueLabel =(int) testingY.at<float>(i,0);
    if (response!=trueLabel){
      err++;
    }
    for (int c=0;c<num_of_classes;c++){
      if(c==response && c==trueLabel){
        TP[c]++;
      }
      if(c==response && c!=trueLabel){
        FP[c]++;
      }
      if(c!=response && c==trueLabel){
        FN[c]++;
      }
      if(c!=response && c!=trueLabel){
        TN[c]++;
      }
    }
    //cout<<response<<endl;
  }
  vector<float> TPR(num_of_classes,0);
  vector<float> FPR(num_of_classes,0);
  vector<float> AUC(num_of_classes,0);
  float AAUC=0.0;
  for(int c = 0;c<num_of_classes;c++){
    TPR[c] = TP[c]==0?0:TP[c]*1.0/(TP[c]*1.0+FN[c]);
    FPR[c] = FP[c]==0?0:FP[c]*1.0/(FP[c]*1.0+TN[c]);
    AUC[c] = TPR[c]*FPR[c]/2.0+ ((TPR[c]+1)*(1-FPR[c]))/2.0;
    AAUC += AUC[c];
  }
  cout<<"AUC="<< AAUC/3<<"  ";
  //cout<<"class err = "<<err*1.0/testingX.rows<<endl;
  //cout<<"fold "<<n<<" err="<<err*1.0/testingX.rows<<endl;
  //free(patternSet);
  //free(base);
  //free(ins);
  //free(smallErrSet);
  //free(largeErrSet);
  
  //free(base);
  //free(ins);
  //free(smallErrSet);
  //free(largeErrSet);
  return err*1.0/testingX.rows;
}

template <class T>
void free(T* p){
  if (p!=NULL){
    delete p;
    p = NULL;
  }
}

int main(int argc, char** argv){
  float error = 0.0;
  analyze_params(argc,argv);
  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *dsa = parser.parse();
  int fold = 7;
  int N = dsa->num_instances();
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
    if (n!=cv_fold)
      continue;
    float err = run(argc,argv, first, last);
    cout<<"fold "<<n<<"  error="<<err<<endl;
    error+=err;
  }
  //cout<<"avg erddror" <<error/fold<<endl;
}
