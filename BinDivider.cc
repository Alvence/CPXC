#include "BinDivider.h"

#include <stdio.h>

#include <iostream>
#include <vector>
#include <cfloat>

#include <arff_value.h>

using namespace std;

BinDivider::~BinDivider(){
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

void BinDivider::init(ArffData* ds, int width){
  this->width = width;
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
  for (int j = 0; j != ds->num_attributes(); j++){
    printf("For attributes %d: max=%5f min=%5f\n",j, maxs->at(j), mins->at(j));
  }
}

BinDivider::BinDivider(){
}

float BinDivider::get_max(int attr_index){
  return 0.0;
}

float BinDivider::get_min(int attr_index){
  return 0.0;
}

int BinDivider::get_index(float val){
  return 0;
}
