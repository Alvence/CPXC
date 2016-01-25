#include "Utils.h"

#include <vector>
#include <iostream>


using namespace std;
template <class T>
void print_vector(vector<T> items){
  cout<<items.size();
  for (int i = 0; i != items.size();i++){
    cout<<" "<<items[i];
  }
  cout<<endl;
}

