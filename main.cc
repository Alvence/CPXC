#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

using namespace cv;
using namespace std;

char* datafile;
char* tempDir;
int classIndex;


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

template <class T>
void print_vector(vector<T> items){
  cout<<items.size();
  for (int i = 0; i != items.size();i++){
    cout<<" "<<items[i];
  }
  cout<<endl;
}


void try_NN(){
  // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, 1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Vec3b green(0,255,0), blue (255,0,0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(0,   0,   0), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}

void translate_input(PatternSet* ps){
  binning_xs.clear();
  for (vector<vector<int> >::iterator it = xs.begin(); it!=xs.end(); it++){
    binning_xs.push_back(ps->translate_input(*it));
  }
}

int main(int argc, char** argv){
  datafile = argv[1];
  tempDir = argv[2];
  classIndex = atoi(argv[3]);
  
  //options for DPM
  int min_sup = atoi(argv[4]);
  int delta = atoi(argv[5]);


  char tempDataFile[32];
  strcpy(tempDataFile,tempDir);
  strcat(tempDataFile,"/binningData.txt");

  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *ds = parser.parse();
  
  printf("Successfully finish reading the data!\n");

  BinDivider* divider= new BinDivider();
  divider->init(ds,5);

  //sava binning temp data file
  printf("saving binning data to %s\n",tempDataFile);
  generate_binning_data(tempDataFile, ds, divider, classIndex);

  int num_of_classes = ds->get_nominal(ds->get_attr(classIndex)->name()).size();
  
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

