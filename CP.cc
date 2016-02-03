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

void Pattern::merge(Pattern p){
    this->union_patterns.push_back(p);
}

bool Pattern::match(vector<int> instance){
  if( isSubset(instance, items)){
    return true;
  }else{
    for (int i = 0; i < this->union_patterns.size(); i++){
        if (isSubset(instance, union_patterns[i].items)){
            return true;
        }
    }
  }
  return false;
}

void Pattern::print(){
  cout<<num_item;
  for (int i = 0; i != items.size();i++){
    cout<<" "<<items[i];
  }
  cout<<endl;
}

float entropy(Pattern &p, vector<vector<int>*>* const xs){
  //TODO
}

float expectedMI(Pattern &p1, Pattern &p2, vector<vector<int>*>* const xs){
  //TODO
}

float MI(Pattern &p1, Pattern &p2, vector<vector<int>*>* const xs){
  //TODO
}

float AMI(Pattern &p1, Pattern &p2, vector<vector<int>*>* const xs){
  float ami = 0.0;
  float mi = 0.0;
  float expMI = 0.0;
  float ent1 = 0.0;
  float ent2 = 0.0;

  //TODO
  
  ami = (mi-expMI)/((ent1>ent2?ent1:ent2)-expMI);
  return ami;
}

void PatternSet::prune_AMI(vector<vector<int>*>* xs){
  vector<vector<float> > matrix;
  //initialize a n*n matrix
  for (int i = 0; i < size; i++){
    vector<float> temp(size,0);
    matrix.push_back(temp);
  }
  for (int i = 0; i < size; i++){
    for (int j = i + 1; j < size; j++){
      matrix[i][j] = AMI(patterns[i],patterns[j],xs);
      matrix[j][i] = matrix[i][j];
    }
  }
  //TODO
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
