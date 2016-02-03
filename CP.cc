#include "CP.h"

#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "Utils.h"

using namespace std;
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

bool Pattern::match(std::vector<int> * instance){
  return match(*instance);
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

float entropy(int count, int N){
  float ent = 0.0;
  float p = count*1.0 / N;
  if (count>0)
    ent = -p * log(p) - (1-p)*log(1-p);
  return ent;
}

float expectedMI(int contigency[][2], int N){
  float expMI = 0.0;
  float E3[2][2];
  float EPLNP[2][2];
  int a[2];
  int b[2];
  for (int i = 0; i < 2; i++){
    a[i] = contigency[i][0] + contigency[i][1];
    b[i] = contigency[0][i] + contigency[1][i];
  }
  for (int i = 0; i <2; i++){
    for (int j = 0; j < 2; j++){
      float ab  = a[i] * b[j];
      E3[i][j] = ab/(N*N) * log(ab/(N*N));
      //std::cout<<i<<","<<j<<":  "<<E3[i][j]<<"\n";
      EPLNP[i][j]=0;
    }
  }
  
  float sumPnij =0.0;
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++){
      sumPnij = 0.0;
      int lower = 1;
      if (lower < a[i]+b[j]-N){
        lower = a[i] + b[j] -N;
      }
      int upper = a[i]<b[j]? a[i]:b[j];
      int nij = lower;
      int temp = N-a[i]-b[j]+nij;
      int X1 = nij<temp?nij:temp;
      int X2 = nij>temp?nij:temp;
      vector<int> nom;
      vector<int> dem;
      if (N-b[j] > X2){
        for (int c = a[i]-nij+1;c <= a[i] ;c++){
          nom.push_back(c);
        }
        for (int c = b[j]-nij+1;c <= b[j] ;c++){
          nom.push_back(c);
        }
        for (int c = X2+1;c <= N-b[j];c++){
          nom.push_back(c);
        }
        for (int c = N-a[i]+1;c <=N ;c++){
          dem.push_back(c);
        }
        for (int c = 1;c <= X1 ;c++){
          dem.push_back(c);
        }
      }else{
        for (int c = a[i]-nij+1;c <= a[i] ;c++){
          nom.push_back(c);
        }
        for (int c = b[j]-nij+1;c <= b[j] ;c++){
          nom.push_back(c);
        }
        for (int c = N-a[i]+1;c <=N ;c++){
          dem.push_back(c);
        }
        for (int c = N-b[j]+1;c <= X2;c++){
          dem.push_back(c);
        }
        for (int c = 1;c <= X1 ;c++){
          dem.push_back(c);
        }
      }//end else
      ////print_vector(nom);
      ////print_vector(dem);
      ////cout<<nom.size()<<" "<<dem.size()<<endl;
      float p0 = 1.0;
      for (int i = 0; i < nom.size();i++){
        p0 = p0 *(nom[i]*1.0/dem[i]);
      }
      p0 = p0/N;

      ////cout<<p0<<endl;
      sumPnij = 0.0;
      EPLNP[i][j] = nij * log(nij*1.0/N)*p0;
      float p1 = p0 *(a[i]-nij)*(b[j]-nij)*1.0/(nij+1)/(N-a[i]-b[j]+nij+1);


      for (nij = lower+1; nij <=upper;nij++){
        sumPnij = sumPnij+p1;
        EPLNP[i][j] = EPLNP[i][j]+nij*log(nij*1.0/N)*p1;
        p1 = p1 * (a[i]-nij)*(b[j]-nij)*1.0/(nij+1)/(N-a[i]-b[j]+nij+1);
      }
    }
  }
  for (int i = 0; i < 2; i++){
    for(int j = 0; j < 2;j++){
      expMI += (EPLNP[i][j]-E3[i][j]);
    }
  }

  cout<<expMI<<endl;
  return expMI;
}

float MI(int contigency[][2], int N){
  int sum_row[2];
  int sum_col[2];
  for (int i = 0; i < 2; i++){
    sum_row[i] = contigency[i][0] + contigency[i][1];
    sum_col[i] = contigency[0][i] + contigency[1][i];
  }

  float mi = 0.0;
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++){
      float p_ij = contigency[i][j]*1.0/N;
      float p_i = sum_row[i]*1.0 / N;
      float p_j = sum_col[j]*1.0 / N;
      if (p_ij!=0 && p_i!=0 && p_j != 0){
        mi += p_ij * log (p_ij/(p_i*p_j));
      }
    }
  }

  return mi;
}

int count(Pattern &p, vector<vector<int>*>* const xs){
  int count = 0;
  for (int i = 0 ; i < xs->size(); i++){
    if (p.match(xs->at(i))){
      count++;
    }
    /*cout<<"p: ";
    p.print();
    cout<<"X: ";
    print_vector(xs->at(i));
    cout<<"match? "<<p.match(xs->at(i))<<endl;
  */
  }
  return count;
}


int count2(Pattern &p1, Pattern &p2, vector<vector<int>*>* const xs){
  int count = 0;
  for (int i = 0 ; i < xs->size(); i++){
    if (p1.match(xs->at(i))&p2.match(xs->at(i))){
      count++;
    }
  }
  return count;
}
float AMI(Pattern &p1, Pattern &p2, vector<vector<int>*>* const xs){
  float ami = 0.0;
  float mi = 0.0;
  float expMI = 0.0;
  float ent1 = 0.0;
  float ent2 = 0.0;

  //number of total instances
  int N = xs->size();
  //number of instances matching p1
  int n1 = count(p1, xs);
  //number of instances matching p2
  int n2 = count(p2, xs);
  //contigency table for p1,~p1,p2,~p2
  int contigency[2][2];
  
  int n12 = count2(p1,p2,xs);
  //number of instances matching p1, p2
  contigency[0][0] = n12;
  //...matching p1, not p2
  contigency[0][1] = n1 - n12;
  //...matching p2, not p1
  contigency[1][0] = n2 - n12;
  //...matching neither p1 nor p2
  contigency[1][1] = N - n1 - n2 + n12;

  ent1 = entropy(n1,N);
  ent2 = entropy(n2,N);
  mi = MI(contigency, N);
  expMI = expectedMI(contigency,N);

/*  p1.print();
  p2.print();

  for (int i = 0; i < xs->size();i++){
    print_vector(xs->at(i));
  }

  //TODO
  for (int i = 0;i<2;i++){
    for (int j = 0; j < 2; j++){
      cout<<contigency[i][j]<<" ";
    }
    cout<<endl;
  }*/
  ////cout <<"ent1="<<ent1<<"  ent2="<<ent2<<"  mi="<<mi<<"  expMI="<<expMI<<endl;
  
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
      if (matrix[i][j] > 0.4){
        cout<<"prune: ("<<i<<","<<j<<"): "<<matrix[i][j]<<endl;
        
      }
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
