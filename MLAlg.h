#ifndef CPXC_MLALG_H
#define CPXC_MLALG_H

#include <vector>

//return test error
float try_SVM(std::vector<std::vector<int>*> *training_X, std::vector<int> *training_Y, std::vector<std::vector<int>* >* testing_X, std::vector<int> * testing_Y);



//return testing error with n-fold cross validation
float try_SVM(std::vector<std::vector<int>*> *training_X, std::vector<int> *training_Y, int fold);
#endif
