#pragma once 

#include <stdio.h>
#include "Global.h"
#include "parameters.h"

class FSout
{
public:

	FSout(char *filename);
	~FSout();

	int isOpen();

	void printSet(int length, int *iset, int support);
	void printPair(int nkey_pat_id, int nclosed_pat_id);

private:

	FILE *out;
};

extern FSout *gpfgout;
extern FSout *gpfcout;
extern FSout *gppair_out;


inline void OutputOneGenerator(int nsupport)
{

	gntotal_generators++;
//	if(gnmax_pattern_len<gnprefix_len)
//		gnmax_pattern_len = gnprefix_len;

	if(goparameters.bresult_name_given)
		gpfgout->printSet(gnprefix_len, gpprefix_itemset, nsupport);
}

inline void OutputOneGenerator(int nitem, int nsupport)
{

	gntotal_generators++;
//	if(gnmax_pattern_len<gnprefix_len+1)
//		gnmax_pattern_len = gnprefix_len+1;

	if(goparameters.bresult_name_given)
	{
		gpprefix_itemset[gnprefix_len] = nitem;
		gpfgout->printSet(gnprefix_len+1, gpprefix_itemset, nsupport);
	}
}

inline void OutputOneClosedPat(int nsupport)
{
	gntotal_closed++;
	if(gnmax_pattern_len<gnprefix_len+1)
		gnmax_pattern_len = gnprefix_len+1;

	if(goparameters.bresult_name_given)
		gpfcout->printSet(gnfull_len, gpfull_itemset, nsupport);

}


inline void	WriteOnePair(int nkey_pat_id, int nclosed_pat_id)
{
	if(goparameters.bresult_name_given)
		gppair_out->printPair(nkey_pat_id, nclosed_pat_id);
}

