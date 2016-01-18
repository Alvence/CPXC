#include <stdio.h>

#include <iostream>
#include <string>

#include <arff_parser.h>
#include <arff_data.h>
#include <arff_value.h>

int main(int argc, char* argv[]){
  ArffParser parser(argv[1]);
  ArffData *ds = parser.parse();

  printf("attributes: %li\n",ds->num_attributes());
  printf("instances: %li\n",ds->num_instances());

  std::vector<std::string> nominals = ds->get_nominal("class");
  for (std::vector<std::string>::iterator it = nominals.begin(); it!=nominals.end();it++){
    std::cout<<*it<<std::endl;
  }

  for (int i = 0; i != ds->num_instances(); i++){
    ArffInstance* x = ds->get_instance(i);
    for (int j = 0; j != ds->num_attributes(); j++){
      ArffValue* v = x->get(j);
      //std::string s = *v;
      printf("instance %d: attribute %d - type=%d  value=\n", i, j, v->type());
    }
  }

  return 0;
}
