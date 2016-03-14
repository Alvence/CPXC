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

	void printKey(int length, int *iset, int support);
	void printClosed(int length, int *iset, int nsupport, int *pclass_distr);
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
		gpfgout->printKey(gnprefix_len, gpprefix_itemset, nsupport);
}

inline void OutputOneGenerator(int nitem, int nsupport)
{

	gntotal_generators++;
//	if(gnmax_pattern_len<gnprefix_len+1)
//		gnmax_pattern_len = gnprefix_len+1;

	if(goparameters.bresult_name_given)
	{
		gpprefix_itemset[gnprefix_len] = nitem;
		gpfgout->printKey(gnprefix_len+1, gpprefix_itemset, nsupport);
	}
}

inline void OutputOneClosedPat(int nsupport, int* pclass_distr)
{
	int nsup_sum, nmax_sup, i;
	nsup_sum = 0;
	nmax_sup = 0;
	for(i=0;i<gnum_of_classes;i++)
	{
		nsup_sum += pclass_distr[i];
		if(nmax_sup<pclass_distr[i])
			nmax_sup = pclass_distr[i];
	}
	if(nsup_sum!=nsupport)
		printf("Error: inconsistent support: %d %d\n", nsup_sum, nsupport);
	if(nsupport-nmax_sup>gndelta && nsupport!=gndb_size)
		printf("Error: this pattern should not be in the output\n");

	gntotal_closed++;
	if(gnmax_pattern_len<gnprefix_len+1)
		gnmax_pattern_len = gnprefix_len+1;

	if(goparameters.bresult_name_given)
		gpfcout->printClosed(gnfull_len, gpfull_itemset, nsupport, pclass_distr);
}


inline void	WriteOnePair(int nkey_pat_id, int nclosed_pat_id)
{
	if(nkey_pat_id==-1 || nclosed_pat_id==-1)
		printf("Error with key id or closed id\n");

	if(goparameters.bresult_name_given)
		gppair_out->printPair(nkey_pat_id, nclosed_pat_id);
}

