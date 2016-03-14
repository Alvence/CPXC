#ifndef CPXC_UTILS_H
#define CPXC_UTILS_H

#include <vector>
#include <iostream>


using namespace std;
template <class T>
void print_vector(vector<T> items){
  cout<<items.size()<<": ";
  for (int i = 0; i != items.size();i++){
    cout<<" "<<items[i];
  }
  cout<<endl;
}

template <class T>
void print_vector(vector<T>* items){
  cout<<items->size()<<": ";
  for (int i = 0; i != items->size();i++){
    cout<<" "<<items->at(i);
  }
  cout<<endl;
}

template <class T>
void free_p(T* p){
  if (p!=NULL){
    delete p;
    p = NULL;
  }
}
#endif
