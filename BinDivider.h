#ifndef CPXC_BINDIVIDER_H
#define CPXC_BINDIVIDER_H

#include <vector>

#include <arff_data.h>

class BinDivider{
private:
    std::vector<std::vector<float>*>* bin_list;
    std::vector<float>* maxs;
    std::vector<float>* mins;
    //indicate whether the corresponding attribute in numeric or not
    std::vector<bool>* flags; 
    int width;
public:
    BinDivider();
    ~BinDivider();

    inline int get_width(){return width;}

    void init(ArffData* ds, int width);
    float get_max(int attr_index);
    float get_min(int attr_index);
    int get_index(float val);
};


#endif
