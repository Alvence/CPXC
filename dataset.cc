#include "dataset.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void Dataset::read_from_file(const char* filename){
  ifstream infile(filename);
  string line;
 
  while (getline(infile,line)){
    cout<<line<<endl;
  }	
}

int main(int argc, char* argv[]){
  Dataset* ds= new Dataset();
  ds->read_from_file(argv[1]);
  delete ds;
}
