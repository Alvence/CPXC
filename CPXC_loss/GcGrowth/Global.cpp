#include <stdlib.h>
#include "Global.h"

//parameter settings
int gnmin_sup;

//some statistic information about the data
int gndb_size;
int gnmax_trans_len;
int gnmax_item_id;

//pattern statistics
int  gntotal_freqitems;
int	gnmax_pattern_len;
int gntotal_singlepath;
int  gntotal_generators;
int gntotal_closed;

//time statistics
double gdtotal_running_time;

//space statistics
double gdused_mem_size;
double gdmax_used_mem_size;
int gntree_init_size;
int gntree_max_size;

//program running statistics
int gntotal_call;
int gngenerator_check_times;


int comp_int_asc(const void *e1, const void *e2)
{
	int n1, n2;
	n1 = *(int*) e1;
	n2 = *(int*) e2;

	if (n1>n2)
		return 1;
	else if (n1<n2)
		return -1;
	else
		return 0;
}


//global varibles
int *gpprefix_itemset;
int gnprefix_len;
int *gpfull_itemset;
int gnfull_len;
int gnglobal_full;

int *gptransaction;
char *gpsort_trans_buf;
int *gpdfs_item_suporder_map;

int gnum_of_newfreqitems;
char *gpitem_bitmap;
char *gpclosed_bitmap;


int *gpsingle_branch;

ITEM_COUNTER *gpdfs_item_counter;