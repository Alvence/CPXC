#include <stdio.h>

#include <iostream>
#include <string>

#include <arff_parser.h>
#include <arff_data.h>
#include <arff_value.h>

#include "BinDivider.h"

using namespace std;

char* datafile;

int get_bin_value(ArffData* ds, string nominal, int attr_index){
  string name = ds->get_attr(attr_index)->name();
  vector<string> nominals = ds->get_nominal(name);
  int value = 0;
  while (value < nominals.size() && nominals[value]!=nominal){
    value++;
  }
  return (attr_index+1)*10 + value;
}

int main(int argc, char** argv){
  datafile = argv[1];
  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *ds = parser.parse();
  
  printf("Successfully finish reading the data!\n");

  BinDivider* divider= new BinDivider();
  divider->init(ds,5);

  for (int i = 0; i != ds->num_instances(); i++){
    ArffInstance* x = ds->get_instance(i);
    printf("\ninstance %d:",i);
    for (int j = 0; j != ds->num_attributes(); j++){
      ArffValue* v = x->get(j);
      //std::string s = *v;
      if (v->type() == FLOAT || v->type() == INTEGER){
        printf("%d ",divider->get_bin_value((float)(*v),j));
      }else if(v->type()==STRING||v->type()==DATE){
        printf("%d",get_bin_value(ds,(string)(*v),j));
      }
    }
  }
  if (divider != NULL){
    delete divider;
    divider = NULL;
  }
  return 0;
}


