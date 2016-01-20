#include "CP.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

template <typename T>
bool isSubset(std::vector<T> A, std::vector<T> B)
{
  std::sort(A.begin(), A.end());
  std::sort(B.begin(), B.end());
  return std::includes(A.begin(), A.end(), B.begin(), B.end());
}

Pattern::Pattern(int n, vector<int> is){
  this->num_item = n;
  this->items = is;
}

bool Pattern::match(vector<int> instance){
  return isSubset(instance, items);
}

void Pattern::print(){
  cout<<num_item;
  for (int i = 0; i != items.size();i++){
    cout<<" "<<items[i];
  }
  cout<<endl;
}

void PatternSet::read(char* file){
  ifstream in;
  in.open(file);
  string line;
  //cout<<file<<endl;
  while (getline(in, line)){
    istringstream iss(line);
    int num_item;
    vector<int> items;
    iss >> num_item;
    for (int i=0; i<num_item; i++){
      int item;
      iss>>item;
      items.push_back(item);
    }
    Pattern newP(num_item, items);
    this->patterns.push_back(newP);
  }
  this->size = patterns.size();
  in.close();
}

vector<int> PatternSet::translate_input(vector<int> input){
  vector<int> newInput;
  for (int i=0; i<patterns.size();i++){
    if(patterns[i].match(input)){
      newInput.push_back(1);
    }else {
      newInput.push_back(0);
    }
  }
  return newInput;
}

void PatternSet::print(){
  for (int i=0; i!=patterns.size(); i++){
    patterns[i].print();
  }
}
