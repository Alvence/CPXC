#include <stdio.h>

#include <iostream>
#include <string>

#include <arff_parser.h>
#include <arff_data.h>

int main(int argc, char* argv[]){
  ArffParser parser(argv[1]);
  ArffData *ds = parser.parse();

  printf("attributes: %li\n",ds->num_attributes());
  printf("instances: %li\n",ds->num_instances());

  std::vector<std::string> nominals = ds->get_nominal("class");
  for (std::vector<std::string>::iterator it = nominals.begin(); it!=nominals.end();it++){
    std::cout<<*it<<std::endl;
  }
  return 0;
}
