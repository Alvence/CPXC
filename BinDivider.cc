#include "BinDivider.h"

#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <set>

#include <arff_value.h>

#include "Utils.h"


using namespace std;

BinDivider::~BinDivider(){
  if(widths!=NULL){
    delete widths;
    widths=NULL;
  }
  if (flags!=NULL){
    delete flags;
    flags=NULL;
  }
  if (bin_list!=NULL){
    for (int i = 0; i != bin_list->size(); i++){
      if (bin_list->at(i) != NULL){
        delete bin_list->at(i);
        bin_list->at(i) = NULL;
      }
    }
    delete bin_list;
    bin_list = NULL;
  }
  if (maxs!=NULL){
    delete maxs;
    maxs = NULL;
  }
  free_p(mins);/*
  if (mins!=NULL){
    delete mins;
    mins = NULL;
  }*/
}
bool isNumeric(ArffValueEnum type){
  return type == INTEGER||
    type == FLOAT||
    type == NUMERIC;
}

void divide(const vector<float> &set, const vector<int> &labels, vector<float> &sub1, vector<int> &labels1, vector<float> &sub2,vector<int> &labels2, float cutting_point, int& k1, int &k2){
  sub1.clear();
  sub2.clear();
  labels1.clear();
  labels2.clear();
  
  std::set<int> l1;
  std::set<int> l2;
  for (int i = 0; i < set.size(); i++){
    if (set[i]<=cutting_point){
      sub1.push_back(set[i]);
      labels1.push_back(labels[i]);
      l1.insert(labels[i]);
    }else{
      sub2.push_back(set[i]);
      labels2.push_back(labels[i]);
      l2.insert(labels[i]);
    }
  }
  k1 = l1.size();
  k2 = l2.size();
}

void divide(const vector<float> &set, const vector<int> &labels, vector<float> &sub1, vector<int> &labels1, vector<float> &sub2,vector<int> &labels2, float cutting_point){
  int k1,k2;
  divide(set,labels,sub1,labels1,sub2,labels2,cutting_point,k1,k2);
}
float entropy(vector<float> set, vector<int> labels, int num_classes){
  vector<int> counts(num_classes,0);
  for(int i = 0; i < labels.size();i++){
    counts[labels[i]]++;
  }
  float ent = 0.0;
  for (int i = 0; i < num_classes; i++){
    float p = counts[i]*1.0/labels.size();
    if (p!=0){
      ent += - p*log2(p);
    } 
  }
  return ent;
}


float entropy(vector<float> set, vector<int> labels, float cutting_point, int num_classes){
  vector<float> sub1;
  vector<float> sub2;
  vector<int> labels1;
  vector<int> labels2;

  divide(set,labels,sub1,labels1,sub2,labels2, cutting_point);
  
  float ent = 0.0;
  ent = labels1.size()*1.0/labels.size()*entropy(sub1,labels1,num_classes) + labels2.size()*1.0/labels.size()*entropy(sub2,labels2,num_classes);  
  return ent;
}
void discretize(vector<float> set, vector<int> labels,vector<float>* dividers, int num_classes){
  vector<float> temp(set);
  vector<float> cutting_points;
  //calculate distinct values in order to find cutting points candidate
  sort(temp.begin(),temp.end());
  ////print_vector(set);
  ////print_vector(labels);
  temp.erase(unique(temp.begin(),temp.end()),temp.end());
  ////print_vector(temp);
  if (temp.size() < 2){
    return;
  }
  for (int i = 0; i < temp.size()-1;i++){
    cutting_points.push_back((temp[i]+temp[i+1])/2.0);
  }
  ////print_vector(cutting_points);
  float cutting_point=0.0;
  float max_gain = -1;
  float marginal_entropy = entropy(set, labels, num_classes);

  ////cout<<"marginal entropy="<<marginal_entropy<<endl;
  for (int i = 0; i < cutting_points.size();i++){
    float conditional_entropy = entropy(set, labels, cutting_points[i], num_classes);
    float gain = marginal_entropy - conditional_entropy;
    ////cout<<(gain-max_gain>1e-7)<<" "<<"cp candidate="<<cutting_points[i]<<"  con-ent="<<conditional_entropy<<"  gain="<<gain<<"  max_gain="<<max_gain<<endl;
    if (gain > max_gain){
      cutting_point = cutting_points[i];
      max_gain = gain;
    }
  }
  vector<float> sub1, sub2;
  vector<int> labels1, labels2;
  int k1=0,k2=0;
  divide(set, labels, sub1, labels1, sub2, labels2, cutting_point, k1, k2);
  int k = k1+k2;
  int N = set.size();
  ////for more about this threshold, see Fayyad&Irani,1993
  float delta = log2(pow(3,k)-2) - (k*marginal_entropy- k1*entropy(sub1,labels1,num_classes)-k2* entropy(sub2, labels2, num_classes));
  float threshold = (log2(N-1) + delta)*1.0/N;
  ////cout<<"cutting point="<<cutting_point<<" k1="<<k1<<"  k2="<<k2<<"  delta="<<delta<<" thre="<<threshold<<endl;
  if (max_gain < threshold){
    return;
  } else{
    dividers->push_back(cutting_point);
    discretize(sub1,labels1,dividers, num_classes);
    discretize(sub2,labels2,dividers, num_classes);
  }
}

int get_nominal_value(ArffData* ds, string nominal, int attr_index){
  string name = ds->get_attr(attr_index)->name();
  vector<string> nominals = ds->get_nominal(name);
  int value = 0;
  while (value < nominals.size() && nominals[value]!=nominal){
    value++;
  }
  return value;
}

void BinDivider::init_minimal_entropy(ArffData* ds, int label_index){
  int dim = ds->num_attributes();
  int num_ins = ds->num_instances();
  int num_bins = 0;
 
  bin_list = new vector<vector<float>*>(dim, NULL);
  flags = new vector<bool>(dim, false);
  //get the label vector
  vector<int> labels(num_ins);
  for (int i = 0; i < num_ins; i++){
    labels[i] = get_nominal_value(ds, (string)(*(ds->get_instance(i)->get(label_index))),label_index);
  }
  for (int j = 0; j < dim; j++){
    if (j == label_index){
      continue;
    }
    vector<float>* dividers = new vector<float>();
    vector<float> set;
    for (int i = 0; i < num_ins; i++){
      ArffValue* v = ds->get_instance(i)->get(j);
      if (isNumeric(v->type()) && !v->missing()){
        flags->at(j) = true;
        set.push_back((float)(*v));
      }
    }
    discretize(set, labels, dividers, ds->get_nominal(ds->get_attr(label_index)->name()).size());
    bin_list->at(j) = dividers;
    cout<<"for att "<<j<<endl;
    print_vector(dividers);
  }
}

void BinDivider::init_equal_width(ArffData* ds, int n){
  int dim = ds->num_attributes();
  int num_ins = ds->num_instances();
  int num_bins = 0;

  //initialize bin_list, maxs and mins
  bin_list = new vector<vector<float>*>(dim, NULL);
  maxs = new vector<float>(dim, FLT_MIN);
  mins = new vector<float>(dim, FLT_MAX);
  flags = new vector<bool>(dim, false);

  //go through the data set to find maxs and mins 
  for (int j = 0; j != dim; j++){
    vector<float> temp;
    for (int i = 0; i != num_ins; i++){
      ArffInstance* x = ds->get_instance(i);
      ArffValue* v = x->get(j);
      if (isNumeric(v->type())&&!v->missing()){
        flags->at(j) = true;
        float value = (float)(*v)*1.0;
        if (value < mins->at(j) ){
          mins->at(j) = value;
        }
        if (value > maxs->at(j) ){
          maxs->at(j)  = value;
        }
        temp.push_back(value);
      }
    }
    if (temp.size()!=0){
      sort(temp.begin(),temp.end());
      vector<float>::iterator last = unique(temp.begin(),temp.end());
      temp.erase(last,temp.end());

      //number of bins = max(n, 2*log(l),l is the number of distinct oberved values, see Spector 1994)
      if (n < 2*log2(temp.size())){
        num_bins = 2*log2(temp.size());
        cout << "attribute "<<j<<" distinct value="<<temp.size();
      }else{
        num_ins = n;
      }
      bin_list->at(j) = new vector<float>(num_bins);
    }
  }
  widths = new vector<float>(dim);
  for (int j = 0; j != ds->num_attributes(); j++){
    if(flags->at(j)){
      num_bins = bin_list->at(j)->size();
      float width = (maxs->at(j)-mins->at(j))/num_bins;
      widths->at(j)=width;
      for (int k =0;  k != num_bins;k++){
        bin_list->at(j)->at(k) = mins->at(j) + width*k;
        //printf("%4f ",bin_list->at(j)->at(k));
      }
      //printf("\n");
      print_vector(bin_list->at(j));
    }
  }
  for (int j = 0; j != ds->num_attributes(); j++){
    printf("For attributes %d: max=%5f min=%5f\n",j, maxs->at(j), mins->at(j));
  }
}

BinDivider::BinDivider(){
}

float BinDivider::get_max(int attr_index){
  return maxs->at(attr_index);
}

float BinDivider::get_min(int attr_index){
  return mins->at(attr_index);
}

int BinDivider::get_bin_value(float val, int attr_index){
  int value = 0;
  vector<float> * bin = bin_list->at(attr_index);
  while(value<bin->size() && bin->at(value)<=val){
    value++;
  }
  return (attr_index+1)*256+value;
}
