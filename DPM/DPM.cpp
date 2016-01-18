#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
//#include <crtdbg.h>

#include <string.h>
#include <time.h>
#include <sys/timeb.h>

#include "DPM.h"
#include "ScanDBMine.h"
#include "FPtree.h"
#include "parameters.h"
#include "fsout.h"
#include "data.h"
#include "inline_routine.h"
#include "PatternSet.h"

FPNODE_BUF gofpnode_buf;
FPCD_BUF gofpcd_buf;
int gnfpcd_page_size;

HEADER_TABLE gpheader_table;

HEADER_TABLE gpdfs_header_array;
int gndfs_header_size;
int gndfs_header_pos;
int* gpdfs_classdistr_array;
int gndfs_classdistr_pos;
int gndfs_classdistr_size;

int *gpdfs_item_classdistr_map;


TAIL_NODE *gpfree_tail_nodes;
TAIL_NODE *gptailnodes_head;
TAIL_NODE *gptailnodes_tail;
int gnum_of_tailnodes;
int gnum_of_tail_pages;
PAT_SET gplocal_fullitems;
int *gplocal_item_sup_map;
int *gplocal_item_sumsup_map;




/*
int main(int argc, char *argv[])
{
	int narg_no;

	if(argc!=5 && argc!=6)
	{
		printf("Usage\n");
		printf("\t%s  data_filename #classes nmin_sup(an obsolute number) ndelta [output_filename]\n", argv[0]);
		return 0;
	}	

	narg_no = 1;
	strcpy(goparameters.szdata_filename, argv[narg_no]);
	narg_no++;

	gnum_of_classes = atoi(argv[narg_no]);
	narg_no++;

	goparameters.nmin_sup = atoi(argv[narg_no]);
	gnmin_sup = goparameters.nmin_sup;
	if(goparameters.nmin_sup<1)
	{
		printf("Please specify a native number for the minimum support threshold\n");
		return 0;
	}
	narg_no++;

	gndelta = atoi(argv[narg_no]);
	narg_no++;

	if(argc==narg_no+1)
	{
		goparameters.bresult_name_given = true;
		goparameters.SetResultName(argv[narg_no]);
	}
	else 
		goparameters.bresult_name_given = false;


	MineFreqGenerators();

	//_CrtDumpMemoryLeaks();

	return gntotal_generators;
}
*/

int dpm(char *filename, char * tempfile, int num_of_classes, int min_sup, int delta)
{
	strcpy(goparameters.szdata_filename, filename);

	gnum_of_classes = num_of_classes;

	goparameters.nmin_sup = min_sup;
	gnmin_sup = goparameters.nmin_sup;

	gndelta = delta;
	goparameters.bresult_name_given = true;
	goparameters.SetResultName(tempfile);

	MineFreqGenerators();

	//_CrtDumpMemoryLeaks();

	return gntotal_generators;
}
void MineFreqGenerators()
{
	FP_NODE *proot;
	HEADER_TABLE pheader_table;
	int *pitem_classsup_map, *pitem_sup_map, ncapacity, i, j, nfreq_item;
	int *pclassdistrs, nmax_class_sup;

	struct timeb start, end;

	ftime(&start);

	if(goparameters.bresult_name_given)
	{
		gpfgout = new FSout(goparameters.szgenerator_filename);
		gpfcout = new FSout(goparameters.szclosed_filename);
		gppair_out = new FSout(goparameters.szpair_filename);
	}
	gpdata = new Data(goparameters.szdata_filename);

	gnmax_pattern_len = 0;
	gntotal_call = 0;
	gdused_mem_size = 0;
	gdmax_used_mem_size = 0;
	gntree_max_size = 0;
	gntotal_generators = 0;
	gntotal_closed = 0;
	gngenerator_check_times = 0;
	gnmap_value = 0;
	gnclosed_mapvalue = 0;

	//count frequent items in original database
	pitem_classsup_map = NULL;
	ncapacity = goDBMiner.ScanDBCountFreqItems(pitem_classsup_map);
	pitem_sup_map = NewIntArray(gnmax_item_id, 0);
	nmax_class_sup = 0;
	for(i=0;i<gnum_of_classes;i++)
	{
		pitem_sup_map[i] = pitem_classsup_map[i*gnum_of_classes+i];
		if(nmax_class_sup<pitem_sup_map[i])
			nmax_class_sup = pitem_sup_map[i];
	}
	for(i=gnum_of_classes;i<gnmax_item_id;i++)
	{
		for(j=0;j<gnum_of_classes;j++)
			pitem_sup_map[i] += pitem_classsup_map[i*gnum_of_classes+j];
	}

	gpprefix_itemset = NewIntArray(gnmax_trans_len);
	gnprefix_len = 0;
	gpfull_itemset = NewIntArray(gnmax_trans_len);
	gnfull_len = 0;

	//enumerate frequent itemsets
	gntotal_freqitems  = 0;
	for(i=gnum_of_classes;i<gnmax_item_id;i++)
	{
		if(pitem_sup_map[i]==gndb_size)
		{
			gpfull_itemset[gnfull_len] = i;
			gnfull_len++;
		}
		else if(pitem_sup_map[i]>=gnmin_sup)
		{
			gntotal_freqitems++;
			nfreq_item = i;
		}			
	}
	gnglobal_full = gnfull_len;

	gplocal_item_sup_map = NewIntArray(gntotal_freqitems);
	gplocal_item_sumsup_map = NewIntArray(gntotal_freqitems);
	gplocal_fullitems.phead = NewPatPage();
	gnum_of_tail_pages = 1;
	gnum_of_tailnodes = 0;
	gptailnodes_head = NewTailNode();

/*	if(gndb_size-nmax_class_sup<=gndelta)
	{
		WriteOnePair(gntotal_generators, gntotal_closed);
		OutputOneGenerator(gndb_size);
		OutputOneClosedPat(gndb_size, pitem_sup_map);
		gnmax_pattern_len = 0;
		delete gpdata;
		DelIntArray(pitem_classsup_map, ncapacity*gnum_of_classes);
		DelIntArray(pitem_sup_map, gnmax_item_id);
	}
	else */
		//WriteOnePair(gntotal_generators, gntotal_closed);
		//OutputOneGenerator(gndb_size);
		//OutputOneClosedPat(gndb_size, pitem_sup_map);

	if(gntotal_freqitems==0)
	{
		gnmax_pattern_len = 0;
		delete gpdata;
		DelIntArray(pitem_classsup_map, ncapacity*gnum_of_classes);
		DelIntArray(pitem_sup_map, gnmax_item_id);
	}
	else if(gntotal_freqitems==1)
	{
		nmax_class_sup = 0;
		for(i=0;i<gnum_of_classes;i++)
		{
			if(nmax_class_sup<pitem_classsup_map[nfreq_item*gnum_of_classes+i])
				nmax_class_sup = pitem_classsup_map[nfreq_item*gnum_of_classes+i];
		}
		if(pitem_sup_map[nfreq_item]-nmax_class_sup<=gndelta)
		{		
			WriteOnePair(gntotal_generators, gntotal_closed);
			OutputOneGenerator(nfreq_item, pitem_sup_map[nfreq_item]);
			gpfull_itemset[gnfull_len++] = nfreq_item;
			OutputOneClosedPat(pitem_sup_map[nfreq_item], &(pitem_classsup_map[nfreq_item*gnum_of_classes]));
			gnfull_len--;
		}
		gnmax_pattern_len = 1;
		delete gpdata;
		DelIntArray(pitem_classsup_map, ncapacity*gnum_of_classes);
		DelIntArray(pitem_sup_map, gnmax_item_id);
	}
	else if(gntotal_freqitems>1)
	{
		pheader_table = NewHeaderTable(gntotal_freqitems);
		pclassdistrs = NewIntArray(gntotal_freqitems*gnum_of_classes);
		gntotal_freqitems = 0;
		for(i=gnum_of_classes;i<gnmax_item_id;i++)
		{
			if(pitem_sup_map[i]>=gnmin_sup && pitem_sup_map[i]<gndb_size)
			{
				pheader_table[gntotal_freqitems].nitem = i;
				pheader_table[gntotal_freqitems].nsupport = pitem_sup_map[i];
				pheader_table[gntotal_freqitems].pconddb = NULL;
				pheader_table[gntotal_freqitems].pclass_distr = &(pclassdistrs[gntotal_freqitems*gnum_of_classes]);
				memcpy(pheader_table[gntotal_freqitems].pclass_distr, &(pitem_classsup_map[i*gnum_of_classes]), sizeof(int)*gnum_of_classes);
				nmax_class_sup = 0;
				for(j=0;j<gnum_of_classes;j++)
				{
					if(nmax_class_sup<pitem_classsup_map[i*gnum_of_classes+j])
						nmax_class_sup = pitem_classsup_map[i*gnum_of_classes+j];
				}
				pheader_table[gntotal_freqitems].nminor_sum = pitem_sup_map[i]-nmax_class_sup;
				gntotal_freqitems++;
			}
		}
		qsort(pheader_table, gntotal_freqitems, sizeof(HEADER_NODE), comp_hnode_freq_des);

		DelIntArray(pitem_classsup_map, ncapacity*gnum_of_classes);
		DelIntArray(pitem_sup_map, gnmax_item_id);

		gpheader_table = pheader_table;

		gpitem_order_map = NewIntArray(gnmax_item_id, -1);
		for(i=0;i<gntotal_freqitems;i++)
			gpitem_order_map[pheader_table[i].nitem] =i;
		gpitem_bitmap = NewCharArray(gntotal_freqitems);
		memset(gpitem_bitmap, 0, sizeof(char)*gntotal_freqitems);
		gpclosed_bitmap = NewCharArray(gntotal_freqitems);
		memset(gpclosed_bitmap, 0, sizeof(char)*gntotal_freqitems);
		gpprefix_orderset = NewUShortArray(gnmax_trans_len);
		gpfull_orderset = NewUShortArray(gnmax_trans_len);
		gpdfs_item_counter = NewItemCounter(gntotal_freqitems);

		Init();
		gopatternset.Init();
		goclosedset.Init();

		gnum_of_newfreqitems = gntotal_freqitems;

		proot = goDBMiner.ScanDBBuildFPtree(pheader_table, gpitem_order_map);
		gntree_init_size = sizeof(FPNODE_BUF)+gofpnode_buf.ntotal_pages*(sizeof(FP_NODE)*FPNODE_PAGE_SIZE+sizeof(FPNODE_PAGE));
		delete gpdata;

		gpdfs_item_suporder_map = NewIntArray(gntotal_freqitems);
		gpdfs_item_classdistr_map = NewIntArray(gntotal_freqitems*gnum_of_classes);

		goFPtree.DepthFGGrowth(proot, pheader_table, gntotal_freqitems, 0);

		DelIntArray(gpdfs_item_suporder_map, gntotal_freqitems);
		DelIntArray(gpdfs_item_classdistr_map, gntotal_freqitems*gnum_of_classes);

		gntree_max_size = sizeof(FPNODE_BUF)+gofpnode_buf.ntotal_pages*(sizeof(FP_NODE)*FPNODE_PAGE_SIZE+sizeof(FPNODE_PAGE));
		
		Destroy();
		gopatternset.Destroy();
		goclosedset.Destroy();
		DelIntArray(gpitem_order_map, gnmax_item_id);
		DelCharArray(gpitem_bitmap, gntotal_freqitems);
		DelCharArray(gpclosed_bitmap, gntotal_freqitems);
		DelUShortArray(gpprefix_orderset, gnmax_trans_len);
		DelUShortArray(gpfull_orderset, gnmax_trans_len);
		DelItemCounter(gpdfs_item_counter, gntotal_freqitems);
		DelHeaderTable(pheader_table, gntotal_freqitems);
		DelIntArray(pclassdistrs, gntotal_freqitems*gnum_of_classes);
	}

	DelIntArray(gpprefix_itemset, gnmax_trans_len);
	DelIntArray(gpfull_itemset, gnmax_trans_len);

	DelIntArray(gplocal_item_sup_map, gntotal_freqitems);
	DelIntArray(gplocal_item_sumsup_map, gntotal_freqitems);
	DelPatSet(&gplocal_fullitems);
	DelTailNodes(gptailnodes_head);

	if(goparameters.bresult_name_given)
	{
		delete gpfgout;
		delete gpfcout;
		delete gppair_out;
	}
	ftime(&end);
	gdtotal_running_time = end.time-start.time+(double)(end.millitm-start.millitm)/1000;

	PrintSummary();
}



void FPtree::DepthFGGrowth(FP_NODE *proot, HEADER_TABLE pheader_table, int num_of_freqitems, int num_of_tail_items)
{
	HEADER_TABLE pnewheader_table;
	FP_NODE *pnewroot, *pfpnode;
	FPNODE_PAGE *pfp_start_page;
	FPCD_PAGE *pfpcd_start_page;
	int k, i, j, nfp_start_pos, nfpcd_start_pos;
	int *pclass_distr, ncount, nmax_class_sup, nlen, nitem, nkey_pat_id, nclosed_pat_id;
	int num_of_tail_comm, num_of_comm_items, num_of_newfreqitems, num_of_newtail_items, num_of_newtail_comm;
	unsigned int norig_closed_mapvalue;
	int norig_full_len;
	bool bnonkey;
	int *pnewclass_distrs;

	gntotal_call++;

	for(k=0;k<num_of_freqitems;k++)
	{
		gpprefix_itemset[gnprefix_len] = pheader_table[k].nitem;
		gpprefix_orderset[gnprefix_len] = gpitem_order_map[pheader_table[k].nitem];
		gpitem_bitmap[gpprefix_orderset[gnprefix_len]] = 1;
		gnprefix_len++;
		gnmap_value = HashAdd1Item(gnmap_value, pheader_table[k].nitem);
		if(gnprefix_len==1)
			gopatternset.CheckMap(k);
		norig_closed_mapvalue = gnclosed_mapvalue;
		norig_full_len = gnfull_len;
		AddOneCommItem(pheader_table[k].nitem);

//if(gnprefix_len>=2 && gpprefix_itemset[0]==3 && gpprefix_itemset[1]==41)
//printf("stop\n");

		nkey_pat_id = gopatternset.InsertKey(pheader_table[k].nsupport, pheader_table[k].nminor_sum);

		if(pheader_table[k].pconddb==NULL)
		{
			if(pheader_table[k].nminor_sum<=gndelta)
			{
				nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport, pheader_table[k].pclass_distr);
				WriteOnePair(gntotal_generators-1, nclosed_pat_id);
			}
		}
		else 
		{
			if(pheader_table[k].pconddb->pnode_link==NULL)
			{
				ncount = pheader_table[k].pconddb->frequency;
				pfpnode = pheader_table[k].pconddb->pparent;
				num_of_comm_items = 0;
				if(ncount==pheader_table[k].nsupport)
				{
					if(pheader_table[k].nminor_sum<=gndelta)
					{
						while(pfpnode!=NULL)
						{
							nitem = pheader_table[pfpnode->nitem_order].nitem;
							AddOneCommItem(nitem);
							num_of_comm_items++;
							pfpnode = pfpnode->pparent;
						}
						num_of_tail_comm = CountTailFullItems(pheader_table[k].pconddb, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, k);
						for(i=0;i<num_of_tail_comm;i++)
						{
							nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
							AddOneCommItem(nitem);
						}
						nclosed_pat_id = goclosedset.InsertClosed(ncount, pheader_table[k].pconddb->pclass_distr);
						WriteOnePair(nkey_pat_id, nclosed_pat_id);
						for(i=0;i<num_of_comm_items+num_of_tail_comm;i++)
						{
							RemoveOneCommItem();
						}
					}
				}
				else if(ncount>=gnmin_sup)
				{
					if(pheader_table[k].nminor_sum<=gndelta)
					{
						nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport, pheader_table[k].pclass_distr);
						WriteOnePair(nkey_pat_id, nclosed_pat_id);
					}
					else
					{

						nmax_class_sup = 0;
						for(i=0;i<gnum_of_classes;i++)
						{
							if(nmax_class_sup<pheader_table[k].pconddb->pclass_distr[i])
								nmax_class_sup = pheader_table[k].pconddb->pclass_distr[i];
						}

						nlen = 0;
						while(pfpnode!=NULL)
						{
							if(gnprefix_len==1)
								gpsingle_branch[nlen++] = pheader_table[pfpnode->nitem_order].nitem;
							else if(pheader_table[pfpnode->nitem_order].nminor_sum>gndelta && ncount!=pheader_table[pfpnode->nitem_order].nsupport)
							{
								if(IsGenerator(pheader_table[pfpnode->nitem_order].nitem, ncount))
									gpsingle_branch[nlen++] = pheader_table[pfpnode->nitem_order].nitem;
							}

							if(ncount-nmax_class_sup<=gndelta)
							{
								nitem = pheader_table[pfpnode->nitem_order].nitem;
								AddOneCommItem(nitem);
								num_of_comm_items++;
							}
							pfpnode = pfpnode->pparent;
						}
						if(nlen>=1)
						{
							if(ncount-nmax_class_sup<=gndelta)
							{
								num_of_tail_comm = CountTailFullItems(pheader_table[k].pconddb, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, k);
								for(i=0;i<num_of_tail_comm;i++)
								{
									nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
									AddOneCommItem(nitem);
								}
								nclosed_pat_id = goclosedset.InsertClosed(ncount, pheader_table[k].pconddb->pclass_distr);
							}
							for(i=0;i<nlen;i++)
							{
								nkey_pat_id = gopatternset.InsertKey(gpsingle_branch[i], ncount, ncount-nmax_class_sup);
								if(ncount-nmax_class_sup<=gndelta)
									WriteOnePair(nkey_pat_id, nclosed_pat_id);
							}
							if(ncount-nmax_class_sup<=gndelta)
							{
								for(i=0;i<num_of_tail_comm;i++)
								{
									RemoveOneCommItem();
								}
							}
						}
						if(ncount-nmax_class_sup<=gndelta)
						{
							for(i=0;i<num_of_comm_items;i++)
							{
								RemoveOneCommItem();
							}
						}
					}
				}
				else
				{
					if(pheader_table[k].nminor_sum<=gndelta)
					{
						nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport, pheader_table[k].pclass_distr);
						WriteOnePair(nkey_pat_id, nclosed_pat_id);
					}
				}
			}
			else
			{
				//count frequent items from AFOPT-tree
				memset(gpdfs_item_suporder_map, 0, sizeof(int)*k);
				memset(&(gpdfs_item_suporder_map[k]), -1, sizeof(int)*(num_of_freqitems+num_of_tail_items-k));
				memset(gpdfs_item_classdistr_map, 0, sizeof(int)*k*gnum_of_classes);
				CountFreqItems(pheader_table, k, gpdfs_item_classdistr_map);
				for(i=0;i<k;i++)
				{
					pclass_distr = &(gpdfs_item_classdistr_map[i*gnum_of_classes]);
					for(j=0;j<gnum_of_classes;j++)
						gpdfs_item_suporder_map[i] += pclass_distr[j];
				}

				num_of_newfreqitems = 0;
				num_of_newtail_items = 0;
				num_of_comm_items = 0;
				for(i=0;i<k;i++)
				{
					if(gpdfs_item_suporder_map[i]==pheader_table[k].nsupport)
					{
						AddOneCommItem(pheader_table[i].nitem);
						num_of_comm_items++;
					}
					else if(gpdfs_item_suporder_map[i]>=gnmin_sup)
					{
						if(pheader_table[i].nminor_sum>gndelta && gpdfs_item_suporder_map[i]!=pheader_table[i].nsupport && (gnprefix_len==1 || IsGenerator(pheader_table[i].nitem, gpdfs_item_suporder_map[i])))
						{
							gpdfs_item_counter[num_of_newfreqitems].nsupport = gpdfs_item_suporder_map[i];
							nmax_class_sup = 0;
							for(j=0;j<gnum_of_classes;j++)
							{
								if(nmax_class_sup<gpdfs_item_classdistr_map[i*gnum_of_classes+j])
									nmax_class_sup = gpdfs_item_classdistr_map[i*gnum_of_classes+j];
							}
							gpdfs_item_counter[num_of_newfreqitems].nminor_sum = gpdfs_item_suporder_map[i]-nmax_class_sup;
						}
						else
							gpdfs_item_counter[num_of_newfreqitems].nsupport = 0;
						gpdfs_item_counter[num_of_newfreqitems].nitem = pheader_table[i].nitem;
						gpdfs_item_counter[num_of_newfreqitems].norder = i;
						num_of_newfreqitems++;
					}
					gpdfs_item_suporder_map[i] = -1;
				}
				if(num_of_newfreqitems>0)
					num_of_tail_comm = CountTailFullItems(pheader_table[k].pconddb, pheader_table[k].nsupport, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, k, true);
				else 
					num_of_tail_comm = CountTailFullItems(pheader_table[k].pconddb, pheader_table[k].nsupport, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, k, false);
				for(i=0;i<num_of_tail_comm;i++)
				{
					nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
					AddOneCommItem(nitem);
				}

				if(pheader_table[k].nminor_sum<=gndelta)
				{
					nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport, pheader_table[k].pclass_distr);
					WriteOnePair(nkey_pat_id, nclosed_pat_id);
				}
				else
				{
					if(num_of_newfreqitems>1)
					{
						qsort(gpdfs_item_counter, num_of_newfreqitems, sizeof(ITEM_COUNTER), comp_item_freq_des);
						for(i=0;i<num_of_newfreqitems;i++)
							gpdfs_item_suporder_map[gpdfs_item_counter[i].norder] = i;
					}

					bnonkey = false;
					i = num_of_newfreqitems-1;
					while(i>=0 && gpdfs_item_counter[i].nsupport==0)
					{
						num_of_newfreqitems--;
						num_of_newtail_items++;						
						i--;
						if(!bnonkey)
							bnonkey = true;
					}

					for(i=k+1;i<num_of_freqitems+num_of_tail_items;i++)
					{
						if(gplocal_item_sumsup_map[i]<pheader_table[k].nsupport && gplocal_item_sumsup_map[i]>=gnmin_sup)
						{
							gpdfs_item_suporder_map[i] = num_of_newfreqitems+num_of_newtail_items;
							gpdfs_item_counter[num_of_newfreqitems+num_of_newtail_items].nitem = pheader_table[i].nitem;
							gpdfs_item_counter[num_of_newfreqitems+num_of_newtail_items].nsupport = 0;
							num_of_newtail_items++;
						}
					}

					if(num_of_newfreqitems==1)
					{
						nkey_pat_id = gopatternset.InsertKey(gpdfs_item_counter[0].nitem, gpdfs_item_counter[0].nsupport, gpdfs_item_counter[0].nminor_sum);

						if(gpdfs_item_counter[0].nminor_sum<=gndelta)
						{
							AddOneCommItem(gpdfs_item_counter[0].nitem);

							num_of_newtail_comm = GetTailFullItems(pheader_table[k].pconddb, gpdfs_item_counter[0].nsupport, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, gpdfs_item_counter[0].norder, gpdfs_item_suporder_map, bnonkey);
							for(i=0;i<num_of_newtail_comm;i++)
							{
								nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
								AddOneCommItem(nitem);
							}
							pclass_distr = &(gpdfs_item_classdistr_map[gpdfs_item_counter[0].norder*gnum_of_classes]);
							nclosed_pat_id = goclosedset.InsertClosed(gpdfs_item_counter[0].nsupport, pclass_distr);
							WriteOnePair(nkey_pat_id, nclosed_pat_id);

							for(i=0;i<num_of_newtail_comm;i++)
							{
								RemoveOneCommItem();
							}
							RemoveOneCommItem();
						}
					}
					else if(num_of_newfreqitems>1)
					{
						pnewheader_table = NewHeaderTable(num_of_newfreqitems+num_of_newtail_items);
						pnewclass_distrs = NewDFSClassDistrs(num_of_newfreqitems*gnum_of_classes);
						for(i=0;i<num_of_newfreqitems;i++)
						{
							pnewheader_table[i].nitem = gpdfs_item_counter[i].nitem;
							pnewheader_table[i].nsupport = gpdfs_item_counter[i].nsupport;
							pnewheader_table[i].nminor_sum = gpdfs_item_counter[i].nminor_sum;
							pnewheader_table[i].pconddb = NULL;
							pnewheader_table[i].pclass_distr = &(pnewclass_distrs[i*gnum_of_classes]);
							memcpy(pnewheader_table[i].pclass_distr, &(gpdfs_item_classdistr_map[gpdfs_item_counter[i].norder*gnum_of_classes]), sizeof(int)*gnum_of_classes);
							
						}
						for(i=num_of_newfreqitems;i<num_of_newfreqitems+num_of_newtail_items;i++)
						{
							pnewheader_table[i].nitem = gpdfs_item_counter[i].nitem;
							pnewheader_table[i].nsupport = gpdfs_item_counter[i].nsupport;
							pnewheader_table[i].nminor_sum = 0;
							pnewheader_table[i].pconddb = NULL;
							pnewheader_table[i].pclass_distr = NULL;						
						}

						gnum_of_newfreqitems = num_of_newfreqitems+num_of_newtail_items;
						pfp_start_page = gofpnode_buf.pcur_page;
						nfp_start_pos = gofpnode_buf.ncur_pos;
						pfpcd_start_page = gofpcd_buf.pcur_page;
						nfpcd_start_pos = gofpcd_buf.ncur_pos;
						pnewroot = BuildNewTreeWTail(pheader_table[k].pconddb, pnewheader_table, gpdfs_item_suporder_map); 

						DepthFGGrowth(pnewroot, pnewheader_table, num_of_newfreqitems, num_of_newtail_items);

						ResetFPBuf(pfp_start_page, nfp_start_pos);
						ResetFPCDBuf(pfpcd_start_page, nfpcd_start_pos);
						DelHeaderTable(pnewheader_table, num_of_newfreqitems+num_of_newtail_items);
						DelDFSClassDistrs(pnewclass_distrs, num_of_newfreqitems*gnum_of_classes);
					}
				}
					
				for(i=0;i<num_of_tail_comm+num_of_comm_items;i++)
				{
					RemoveOneCommItem();
				}
			}
		}
		gnprefix_len--;
		gpitem_bitmap[gpprefix_orderset[gnprefix_len]] = 0;
		gnmap_value = HashRemove1Item(gnmap_value, pheader_table[k].nitem);

		RemoveOneCommItem();
		gnclosed_mapvalue = norig_closed_mapvalue;
		if(gnfull_len!=norig_full_len)
			printf("Error with full prefix length\n");
	}
}


bool IsGenerator(int nitem, int nsupport)
{
	bool bisgenerator;

	gngenerator_check_times++;

	gpitem_bitmap[gpitem_order_map[nitem]] = 1;

	bisgenerator = gopatternset.IsGenerator(nitem, nsupport);

	gpitem_bitmap[gpitem_order_map[nitem]] = 0;

	return bisgenerator;
}


void PrintSummary()
{
	printf("#generators: %d\t#closed itemsets: %d\n", gntotal_generators, gntotal_closed);
	printf("#recursive calls: %d\n", gntotal_call);
	printf("#tail nodes: %d\t#tail pages: %d\n", gnum_of_tailnodes, gnum_of_tail_pages);

	if(gdused_mem_size!=0)
		printf("Error with memory: %.2f are not released\n", gdused_mem_size);
	/*
	FILE *fp_sum;

	fp_sum = fopen("DPM.sum.txt", "a+");
	if(fp_sum == NULL)
	{
		printf("Error[PrintSummary]: cannot open file DPM.sum.txt\n");
		return;
	}
	fprintf(fp_sum, "DPM-delta %s ", goparameters.szdata_filename);
	fprintf(fp_sum, "%f %d\t", (double)goparameters.nmin_sup*100/gndb_size, gndelta);
	fprintf(fp_sum, "%d %d %d %d\t", gnmax_pattern_len, gntotal_generators, gntotal_closed, gngenerator_check_times);
	fprintf(fp_sum, "%.2f\t", gdtotal_running_time);
	fprintf(fp_sum, "%d\t", gntotal_call);
	fprintf(fp_sum, "%.2fMB %.2fMB %.2fMB\t", gdmax_used_mem_size/(1<<20), (double)gntree_init_size/(1<<20), (double)gntree_max_size/(1<<20));

	fprintf(fp_sum, "\n");
	fclose(fp_sum);*/

}

