#ifndef CPXC_CP_H
#define CPXC_CP_H

#define JACCARD_SIMILARITY_THRESHOLD 0.9

#include <vector>

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

  inline int get_num_item(){return num_item;}
};

class PatternSet{
private:
  std::vector<Pattern> patterns;
  int size;
public:
  void print();
  void read(char* file);
  void prune_AMI(vector<vector<int>*>* xs, float threshold);
 
  //filter by 3 strategies used in CPXC
  void filter();
  vector<int> translate_input(vector<int>);
  inline int get_size(){return size;}
  inline vector<Pattern> get_patterns(){return patterns;}
};

#endif
