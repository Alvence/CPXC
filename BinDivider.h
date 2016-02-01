#ifndef CPXC_BINDIVIDER_H
#define CPXC_BINDIVIDER_H

#include <vector>
#include <string>

#include <arff_data.h>

enum DiscretizationMethod{EQUAL_WIDTH=0, MINIMAL_ENTROPY};

class BinDivider{
private:
    std::vector<std::vector<float> *>* bin_list;
    std::vector<float>* maxs;
    std::vector<float>* mins;
    //indicate whether the corresponding attribute in numeric or not
    std::vector<bool>* flags;
    std::vector<float>* widths;
public:
    BinDivider();
    ~BinDivider();

    void init_equal_width(ArffData* ds, int n);
    void init_minimal_entropy(ArffData* ds, int label_index);
    float get_max(int attr_index);
    float get_min(int attr_index);
    int get_bin_value(float val, int attr_index);
};


#endif
