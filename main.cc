#include <stdio.h>

#include <iostream>
#include <string>

#include <arff_parser.h>
#include <arff_data.h>
#include <arff_value.h>

#include "BinDivider.h"

using namespace std;

char* datafile;

int main(int argc, char** argv){
  datafile = argv[1];
  //open data file with arff format
  ArffParser parser(datafile);
  //parse the data
  ArffData *ds = parser.parse();
  
  printf("Successfully finish reading the data!\n");

  BinDivider* divider= new BinDivider();
  divider->init(ds,5);

  if (divider != NULL){
    delete divider;
    divider = NULL;
  }
  return 0;
}
