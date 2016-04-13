#include "fsout.h"

#pragma warning (disable:4996)

FSout *gpfgout;
FSout *gpfcout;
FSout *gppair_out;


FSout::FSout(char *filename)
{
  out = fopen(filename,"wt");
}

FSout::~FSout()
{
  if(out) fclose(out);
}

int FSout::isOpen()
{
  if(out) return 1;
  else return 0;
}

void FSout::printSet(int length, int *iset, int support)
{
	fprintf(out, "%d ", length);
	for(int i=0; i<length; i++) 
		fprintf(out, "%d ", iset[i]);
	fprintf(out, "%d\n", support);
}

void FSout::printPair(int nkey_pat_id, int nclosed_pat_id)
{
	fprintf(out, "%d %d\n", nkey_pat_id, nclosed_pat_id);
}
