#ifndef CPXC_CP_H
#define CPXC_CP_H

#define JACCARD_SIMILARITY_THRESHOLD 0.9

#include <vector>
#include <fstream>

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
  void print(std::fstream& fs);
  void merge(Pattern p);

  inline std::vector<int> get_items(){ return items;}
  inline int get_num_item(){return num_item;}
};

class PatternSet{
private:
  std::vector<Pattern> patterns;
  int size;
public:
  std::vector<int> MGs;
  std::vector<std::vector<int> > MGSet;
  void print();
  void read(char* file);
  void prune_AMI(vector<vector<int>*>* xs, float threshold);
 
  //filter by 3 strategies used in CPXC
  void filter(vector<vector<int>*>* const xs,vector< vector<vector<int> *> *>* labelledXs ,int num_attr, float suppRatio);
  vector<int> translate_input(vector<int>);
  inline int get_size(){return patterns.size();}
  inline vector<Pattern> get_patterns(){return patterns;}
  void save(char * filename);

  void MG();
};

#endif
