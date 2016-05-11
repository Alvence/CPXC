#ifndef CPXC_CP_H
#define CPXC_CP_H

#include <vector>
#include "BinDivider.h"
using namespace std;

class Pattern{
private:
  int num_item;
  std::vector<int> items;
  std::vector<Pattern > union_patterns;
public:
  Pattern(int n, std::vector<int> is);

  bool match(std::vector<int> instance);
  bool match(std::vector<int> * instance);
  void print();
  void merge(Pattern p);
  float distance(Pattern p, BinDivider* divider);

  inline int get_num_item(){return num_item;}
};

class PatternSet{
private:
  std::vector<Pattern> patterns;
  int size;
  BinDivider* divider;
public:
  void print();
  void read(char* file, BinDivider* div);
  void prune_AMI(vector<vector<int>*>* xs, float threshold, float sigma);
  
  vector<int> translate_input(vector<int>);
  inline int get_size(){return size;}
};

#endif
