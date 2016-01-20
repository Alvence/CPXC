#include "BinDivider.h"

#include <stdio.h>

#include <iostream>
#include <vector>
#include <cfloat>

#include <arff_value.h>

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
  if (mins!=NULL){
    delete mins;
    mins = NULL;
  }
}

bool isNumeric(ArffValueEnum type){
  return type == INTEGER||
    type == FLOAT||
    type == NUMERIC;
}

void BinDivider::init(ArffData* ds, int n){
  this->num_bins = n;
  int dim = ds->num_attributes();

  //initialize bin_list, maxs and mins
  bin_list = new vector<vector<float>*>(dim, NULL);
  maxs = new vector<float>(dim, FLT_MIN);
  mins = new vector<float>(dim, FLT_MAX);
  flags = new vector<bool>(dim, false);

  //go through the data set to find maxs and mins 
  for (int i = 0; i != ds->num_instances(); i++){
    ArffInstance* x = ds->get_instance(i);
    for (int j = 0; j != ds->num_attributes(); j++){
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
      }
    }
  }
  widths = new vector<float>(dim);
  for (int j = 0; j != ds->num_attributes(); j++){
    if(flags->at(j)){
      float width = (maxs->at(j)-mins->at(j))/num_bins;
      widths->at(j)=width;
      bin_list->at(j) = new vector<float>(num_bins);
      for (int k =0;  k != num_bins;k++){
        bin_list->at(j)->at(k) = mins->at(j) + width*k;
        //printf("%4f ",bin_list->at(j)->at(k));
      }
      //printf("\n");
    }
  }
  //for (int j = 0; j != ds->num_attributes(); j++){
    //printf("For attributes %d: max=%5f min=%5f\n",j, maxs->at(j), mins->at(j));
  //}
  
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
  while(value<num_bins && bin->at(value)<=val){
    value++;
  }
  return (attr_index+1)*10+value;
}
