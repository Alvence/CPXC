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


void LocalClassifier::statBinaryCase(cv::Mat& samples, cv::Mat &labels, float & acc, float& fscore){
  float precision, recall;
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;
  int err=0;
  for(int i = 0; i < samples.rows;i++){
    int rep = (int) predict(samples.row(i));
    int trueLabel = (int) labels.at<float>(i,0);
    if(trueLabel == 1 && rep == 1){
      tp++;
    }else if(trueLabel == 0 && rep == 1){
      fp++;
      err++;
    }
    if(trueLabel == 0 && rep == 0){
      tn++;
    }
    if(trueLabel == 1 && rep == 0){
      fn++;
      err++;
    }
  }
  acc = 1 - err*1.0/samples.rows;
  precision = tp*1.0 / (fp+tp);
  recall = tp*1.0 / (tp+fn);
  float s1 = 2 * (precision*recall)*1.0/(precision + recall);
  precision = tn*1.0 / (fn+tn);
  recall = tn*1.0 / (tn+fp);
  float s2 = 2 * (precision*recall)*1.0/(precision + recall);
  fscore = (s1+s2)/2;
}

float LocalClassifier::predict(Mat sample, Mat& probs){
  if (sample.rows>1){
    return 0;
  }

  if (singleClass >= 0){
    probs.create(1,num_classes,CV_32FC1);
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

float CPXC::predict1(Mat sample, vector<int>* bin_ins, Mat &probs){
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
    return defaultClassifier->predict(sample,probs);
  }
  probs.create(1,num_of_classes,CV_32FC1);

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


float CPXC::TER(Ptr<NormalBayesClassifier> base, Mat &xs, Mat &ys, std::vector<std::vector<int>* >* bin_xs){
    float totErrD = 0;
    float totErrM = 0;

    vector<int> statCov;

    for(int i =0; i < bin_xs->size(); i ++){
      float errD;
      float errM;
      Mat sample = xs.row(i);
      Mat probs;
      Mat result;
      base->predictProb(sample, result, probs);

      float V = 0;
      for (int c = 0; c < num_of_classes; c++){
        V+=probs.at<float>(0,c);
      }
      if(V==0){
        continue; 
      }
      int res = (int) result.at<float>(0,0);
      float prob = probs.at<float>(0,res)/V;
      if (fabs(res - ys.at<float>(i,0))>1e-7){
        errD = 1;
      } else{
        errD = 1 - prob;
      }

      totErrD+=errD;

      Mat probs1;
      vector<int> *matches = getMatches(bin_xs->at(i));
      statCov.push_back(matches->size());
      if(matches->size()==0){
        delete matches;
        continue;
      }
      float response = predict(sample,matches,probs1);
      V = 0;
      for (int c = 0; c < num_of_classes; c++){
        V+=probs1.at<float>(0,c); 
      }
      prob = probs1.at<float>(0,response)/V;
    
      if (fabs(response - ys.at<float>(i,0))>1e-7){
        errM = 1;
      } else{
        errM = 1 - prob;
      }

      totErrM = totErrM + fabs(errD-errM);
      delete matches;
    }
    //print_cover(statCov);
    if(fabs(totErrD)<1e-8){
      return 0;
    }else{
      return totErrM / totErrD;
    }
}
void CPXC::print_pattern_cover(vector<vector<int>* >* xs){
  vector<int> stat(classifiers->size(),0);
  for(int i = 0; i < xs->size();i++){
    for(int j = 0; j < classifiers->size();j++){
      if(classifiers->at(j)->pattern.match(xs->at(i))){
        stat[j]++;
      }
    }
  }
  cout<<"Pattern Coverage for Patterns:"<<endl;
  print_vector(stat);
}

void CPXC::print_cover(vector<int> statCov){
    int countZ = 0;
    int countV = 0;
    for(int i = 0; i < statCov.size();i++){
      if(statCov[i]==0){
        countZ++;
      }
      countV+= statCov[i];
    }
    cout<<"count zero = "<<countZ<<" avarage cov="<<countV*1.0/statCov.size()<<endl;
}

CPXC generate_new(CPXC* origin, vector<int> PS){
  CPXC res;
  res.num_of_classes = origin->num_of_classes;
  res.defaultClassifier = origin->defaultClassifier;
  res.classifiers = new vector<LocalClassifier*>();
  
  for(int i = 0; i < PS.size(); i++){
    res.classifiers->push_back(origin->classifiers->at(PS[i]));
  }

  return res;
}
float CPXC::obj(Ptr<NormalBayesClassifier> base, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>* >* bin_xs){
  return TER(base, xs, ys, bin_xs);
}

CPXC CPXC::optimize(int k,Ptr<NormalBayesClassifier> base, cv::Mat &xs, cv::Mat &ys, std::vector<std::vector<int>* >* bin_xs){
  vector<int> PS;
  vector<int> NPS;
  for(int i = 0; i < classifiers->size();i++){
    NPS.push_back(i);
  }

  while(PS.size()<k){
    CPXC curC = generate_new(this,PS);
    float objV = curC.obj(base, xs, ys, bin_xs);
    float curObj = 0;
    int cur = -1;
    for(int i = 0; i < NPS.size();i++){
      vector<int> tempPS = PS;
      tempPS.push_back(NPS[i]);
      CPXC tempC = generate_new(this, tempPS);
      float tempO =  tempC.obj(base, xs, ys, bin_xs);
      if(curObj < tempO){
        cur = i;
        curObj = tempO;
      }
    }
    
    if(cur!=-1){
      PS.push_back(NPS[cur]);
      NPS.erase(NPS.begin()+cur);
    }
    int iPS = -1;
    int iNPS = -1;
    for (int i = 0; i < PS.size();i++){
      for(int j = 0; j < NPS.size();j++){
        vector<int> tempPS = PS;
        int vj =NPS[j];
        tempPS.erase(tempPS.begin()+i);
        tempPS.push_back(vj);
        CPXC tempC = generate_new(this, tempPS);
        if(curObj < tempC.obj(base, xs, ys, bin_xs)){
          iPS = i;
          iNPS = j;
        }
      }
    }
    if((curObj - objV)/objV < 0.001){
      break;
    }else if (iPS!=-1){
      int v1 = PS[iPS];
      int v2 = NPS[iNPS];
      PS.erase(PS.begin()+iPS);
      NPS.erase(NPS.begin()+iNPS);
      PS.push_back(v2);
      NPS.push_back(v1);
    }
  }
  return generate_new(this,PS);
}


