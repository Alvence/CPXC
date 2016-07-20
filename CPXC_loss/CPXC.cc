#include "CPXC.h"

#include <cmath>
#include <fstream>
#include "Utils.h"
using namespace cv;
using namespace std;

float LocalClassifier::train(Mat & trainingX, Mat & trainingY){
  NBC->train(trainingX,ROW_SAMPLE, trainingY);
}

float LocalClassifier::predict(Mat sample){
  if (sample.rows>1){
    return 0;
  }
  if (singleClass >= 0){
    return singleClass;
  }
  Mat result;
  Mat probs;
  NBC->predictProb(sample,result,probs);
  return result.at<float>(0,0);
}

float LocalClassifier::predict(Mat sample, Mat& probs){
  if (sample.rows>1){
    return 0;
  }

  if (singleClass >= 0){
    probs=Mat::zeros(1,num_classes,CV_32FC1);
    probs.at<float>(0,singleClass) = 1.0;
    return singleClass;
  }
  Mat result;
  NBC->predictProb(sample,result,probs);

  float total = 0.0;
  for (int c=0; c < probs.cols;c++){
    total+= probs.at<float>(0,c);
  }
  for (int c=0; c < probs.cols;c++){
    probs.at<float>(0,c)/=total;
  }

  return result.at<float>(0,0);
}

void CPXC::save(char* filename){
  fstream fs;
  fs.open(filename,fstream::in);

  for (int i=0; i < classifiers->size();i++){
    if (fabs(classifiers->at(i)->weight)<1e-6){
      continue;
    }
    fs<<"pattern:  ";
    classifiers->at(i)->pattern.print(fs);
    fs<<"weight = "<< classifiers->at(i)->weight<<endl<<endl;
  }

  fs.close();
}
/*
void CPXC::filter(PatternSet* patterns, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>*>* mds){
  for (int i = 0; i < mds->size(); i++){
    vector<int>* md = mds->at(i);
    Mat trainingX = Mat::zeros(md->size(),xs.cols, CV_32FC1);
    Mat trainingY = Mat::zeros(md->size(),1,CV_32SC1);
    for (int j = 0; j < md->size(); j++){
      int index = md->at(j);
      for (int col = 0; col < trainingX.cols; col++){
        trainingX.at<float>(j,col)=xs.at<float>(index,col);
      }
      trainingY.at<float>(j,0)=ys.at<float>(index,0);
    }
  } 
}
*/
void CPXC::train(PatternSet* patterns, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>*>* mds, LocalClassifier *baseClassifier, int num_of_classes){
  int num_ins = xs.rows;
  vector<int> flags(num_ins,0);
  int ccc = 0;
  classifiers->clear();
  for (int i = 0; i < mds->size(); i++){
    vector<int>* md = mds->at(i);
    if (md==NULL || md->size()<1){
      LocalClassifier* cf = new LocalClassifier();
      cf->weight = 0;
      cf->singleClass = 1;
      cf->num_classes = num_of_classes;
      classifiers->push_back(cf);
      continue;
    } 
    LocalClassifier* cf = new LocalClassifier();
    cf->num_classes = num_of_classes;
    Mat trainingX = Mat::zeros(md->size(),xs.cols, CV_32FC1);
    Mat trainingY = Mat::zeros(md->size(),1,CV_32SC1);
    for (int j = 0; j < md->size(); j++){
      int index = md->at(j);
      for (int col = 0; col < trainingX.cols; col++){
        trainingX.at<float>(j,col)=xs.at<float>(index,col);
      }
      trainingY.at<float>(j,0)=ys.at<float>(index,0);
      //xs.row(index).copyTo(trainingX.row(j));
      //ys.row(index).copyTo(trainingY.row(j));
      flags[index] = 1;
      cf->classes.insert((int)trainingY.at<float>(j,0));
    }
    ////for (int k=0;k < md->size();k++){
    ////  cout<< md->at(k)<<":"<<trainingY.at<float>(k,0)<<":"<<ys.at<float>(md->at(k),0)<<"  ";
    ////}cout<<endl;
    

    cf->pattern = patterns->get_patterns().at(i);

    //check whether it is single class
    if (cf->classes.size()>1){

      cf->train(trainingX,trainingY);
      //cout<<cf->pattern<<endl;
      //cout<<"complete training for pattern" << i<<endl;
    }else{
      cf->singleClass = *(cf->classes.begin());
    }
    float errReduction = 0.0;
//cout<<"classes num="<<cf->classes.size()<<endl;
    for (int j = 0; j < trainingX.rows; j++){
      Mat probs1 = Mat::zeros(1,num_of_classes,CV_32FC1);
      Mat probs2 = Mat::zeros(1,num_of_classes,CV_32FC1);
//cout<<probs1.size()<<"   vs "<<probs2.size()<<endl;
      int res1 = (int)cf->predict(trainingX.row(j),probs1);
      int res2 = (int)baseClassifier->predict(trainingX.row(j),probs2);
      float norm1 = 0.0;
      float norm2 = 0.0;
     
//cout<<probs1.size()<<"   vs "<<probs2.size()<<endl;
//cout<<"res1="<<res1<<"     res2="<<res2<<endl;

      float prob1 = 0, prob2 = 0;
      for (int c = 0; c < probs1.cols;c++){
        norm1+=probs1.at<float>(0,c);
        if (prob1 < probs1.at<float>(0,c)){
          prob1 = probs1.at<float>(0,c);
        }
      }
      for (int c = 0; c < probs2.cols;c++){
        norm2+=probs2.at<float>(0,c);
        if (prob2 < probs2.at<float>(0,c)){
          prob2 = probs2.at<float>(0,c);
        }
      }

      float err1 = 0.0, err2 = 0.0;

      int label = (int)trainingY.at<float>(j,0);
      if(res1 == label){
        err1 = 1 -  prob1/norm1;
      }else{
        err1 = 1.0;
      }
      if(res2 == label){
        err2 = 1 -  prob2/norm2;
      }else{
        err2 = 1.0;
      }

//cout<<err1<<"   vs "<<err2<<endl;

    errReduction += fabs(err1-err2);
    //  errReduction += err1-err2;
//cout<<"for data "<<j<<" err1="<<err1<<"  err2="<<err2<<"   err1-err2="<<err1-err2<<endl;
    }
    errReduction/= md->size();

//cout<<"for pattern "<<i<<"  aer="<<errReduction<<endl;
  
    if (errReduction < 0.1){
      ccc++;
      cf->weight = 0;
    } else{
      //cf->pattern = &((patterns->get_patterns())[i]);
      cf->weight = errReduction;
    }
    classifiers->push_back(cf);
  }
  //cout<<"ccc="<<ccc<<endl;
  Mat dtrainingX;
  Mat dtrainingY;
  int j=0;
  for (int i = 0; i < num_ins;i++){
    if(flags[i]!=1){
      dtrainingX.push_back(xs.row(i));
      dtrainingY.push_back(ys.row(i));
    }
  }
  if (dtrainingX.rows>10){
    defaultClassifier = new LocalClassifier();
    defaultClassifier->train(dtrainingX,dtrainingY);
  }
  else{
    defaultClassifier = baseClassifier;
  }
}

vector<int>* CPXC::getMatches(vector<int>* ins){
  vector<int>*res = new vector<int>();
  for (int i =0; i< classifiers->size(); i++){
    if (classifiers->at(i)->pattern.match(ins) ){
      res->push_back(i);
    }
  }
  return res;
}

float CPXC::predict1(Mat sample, vector<int>* bin_ins, Mat&probs){
  vector<int> *matches = getMatches(bin_ins);
  float response = predict(sample,matches,probs);
  delete matches;
  return response;
}

float CPXC::predict1(cv::Mat sample, vector<int>* bin_ins){
  Mat probs;
  return predict1(sample, bin_ins, probs);
}

float CPXC::predict(Mat sample, vector<int>* matches){
  if (matches->size() == 0){
    return defaultClassifier->predict(sample);
  }
  vector<float> votes(num_of_classes,0);
  bool flag = true;
  for (int i =0; i < matches->size(); i++){
    float response = classifiers->at(matches->at(i))->predict(sample);
    votes[(int)response] += classifiers->at(matches->at(i))->weight;
    if (fabs(classifiers->at(matches->at(i))->weight)>1e-6){
      flag =false;
    }
  }
  if (flag){
    return defaultClassifier->predict(sample);
  }
  int label;
  float maxvote=-1;
  for (int i = 0; i < num_of_classes;i++){
    if (votes[i]>maxvote){
      label = i;
      maxvote = votes[i];
    }
  }
  return label;
}
float CPXC::predict(Mat sample, vector<int>* matches, Mat &probs){
  if (matches->size() == 0){
    return defaultClassifier->predict(sample);
  }
  probs = Mat::zeros(1,num_of_classes,CV_32FC1);
  vector<float> votes(num_of_classes,0);
  bool flag = true;
  for (int i =0; i < matches->size(); i++){
    float response = classifiers->at(matches->at(i))->predict(sample);
    votes[(int)response] += classifiers->at(matches->at(i))->weight;
    if (fabs(classifiers->at(matches->at(i))->weight)>1e-6){
      flag =false;
    }
  }
  if (flag){
    return defaultClassifier->predict(sample,probs);
  }
  int label;
  float maxvote = -1;
  float totalVote = 0;
  for (int i = 0; i < num_of_classes;i++){
    totalVote+= votes[i];
  }
  for (int i = 0; i < num_of_classes;i++){
    if (votes[i]>maxvote){
      label = i;
      maxvote = votes[i];
    }
    probs.at<float>(0,i) = votes[i]/totalVote;
  }
  return label;
}


float TER(Ptr<NormalBayesClassifier> base, Mat &xs, Mat &ys){
  
}
