#ifndef CPXC_CPXC_H
#define CPXC_CPXC_H

#include <opencv2/core/core.hpp>
class CPXC{
public:
  Pattern pattern;
  float weight;

  float train(cv::Mat &samples, cv::Mat &labels);
  float predict(cv::Mat samples);
};

#endif
