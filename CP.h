#ifndef CPXC_CP_H
#define CPXC_CP_H

#include <vector>

using namespace std;

class Pattern{
private:
  int num_item;
  std::vector<int> items;
public:
  Pattern(int n, std::vector<int> is);

  bool match(std::vector<int> instance);
  void print();

  inline int get_num_item(){return num_item;}
};

class PatternSet{
private:
  std::vector<Pattern> patterns;
  int size;
public:
  void print();
  void read(char* file);
  
  vector<int> translate_input(vector<int>);
  inline int get_size(){return size;}
};

#endif
