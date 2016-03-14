#include "fsout.h"

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

void FSout::printKey(int length, int *iset, int support)
{
	fprintf(out, "%d ", length);
	for(int i=0; i<length; i++) 
		fprintf(out, "%d ", iset[i]);
//	fprintf(out, "%d", support);
	fprintf(out, "\n");

}

void FSout::printClosed(int length, int *iset, int nsupport, int* pclass_distr)
{
	int i;

	fprintf(out, "%d ", length);
	for(i=0; i<length; i++) 
		fprintf(out, "%d ", iset[i]);
	fprintf(out, "   ");
//	fprintf(out, "%d ", nsupport);
	for(i=0;i<gnum_of_classes;i++)
		fprintf(out, "%d ", pclass_distr[i]);
	fprintf(out, "\n");
}


void FSout::printPair(int nkey_pat_id, int nclosed_pat_id)
{
	fprintf(out, "%d %d\n", nkey_pat_id, nclosed_pat_id);
}
