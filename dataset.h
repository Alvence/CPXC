#ifndef CPXC_DATASET_H
#define CPXC_DATASET_H

#include<string>
#include<vector>
using namespace std;

class Instance{
public:
  vector<double> attributes;
  int label;
};

class Dataset{
public:
  string name;
  int num_of_instances;
  int num_of_attributes;
  vector<string> names_of_attributes;
  vector<Instance*> instances;

  void read_from_file(const char* filename);
  void write_to_file(const char* filename);

  ~Dataset(){
    for (vector<Instance*>::iterator it = instances.begin(); it!=instances.end(); it++){
      Instance* instance = (*it);
      if (instance!=NULL){
        delete instance;
      }
    }    
  }
};

#endif
