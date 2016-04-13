#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#pragma warning (disable:4996)

#include <string>
#include <time.h>
#include <sys/timeb.h>

#include "ScanDBMine.h"
#include "FPtree.h"
#include "parameters.h"
#include "fsout.h"
#include "data.h"
#include "inline_routine.h"
#include "PatternSet.h"

FPNODE_BUF gofpnode_buf;
HEADER_TABLE gpheader_table;

HEADER_TABLE gpdfs_header_array;
int gndfs_header_size;
int gndfs_header_pos;

TAIL_NODE *gpfree_tail_nodes;
TAIL_NODE *gptailnodes_head;
TAIL_NODE *gptailnodes_tail;
PAT_SET gplocal_fullitems;
int *gplocal_item_sup_map;
int *gplocal_item_sumsup_map;


bool IsGenerator(int nitem, int nsupport);

/*
int main(int argc, char *argv[])
{

	if(argc!=3 && argc!=4)
	{
		printf("Usage\n");
		printf("\t%s  data_filename  nmin_sup(an obsolute number) [output_filename]\n", argv[0]);
		return 0;
	}	

	strcpy(goparameters.szdata_filename, argv[1]);
	goparameters.nmin_sup = atoi(argv[2]);
	gnmin_sup = goparameters.nmin_sup;
	if(goparameters.nmin_sup<1)
	{
		printf("Please specify a native number for the minimum support threshold\n");
		return 0;
	}
	if(argc==4)
	{
		goparameters.bresult_name_given = true;
		goparameters.SetResultName(argv[3]);
	}
	else 
		goparameters.bresult_name_given = false;


	MineFreqGenerators();

	_CrtDumpMemoryLeaks();

	return gntotal_generators;
}*/

int gcgrowth(char *filename, char * tempfile, int min_sup)
{
	strcpy(goparameters.szdata_filename, filename);


	goparameters.nmin_sup = min_sup;
	gnmin_sup = goparameters.nmin_sup;

	
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
	int *pitem_sup_map, ncapacity, i, nfreq_item;

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
	pitem_sup_map = NULL;
	ncapacity = goDBMiner.ScanDBCountFreqItems(pitem_sup_map);

	gpprefix_itemset = NewIntArray(gnmax_trans_len);
	gnprefix_len = 0;
	gpfull_itemset = NewIntArray(gnmax_trans_len);
	gnfull_len = 0;

	//enumerate frequent itemsets
	gntotal_freqitems  = 0;
	for(i=0;i<gnmax_item_id;i++)
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
	gptailnodes_head = NewTailNode();
	
	WriteOnePair(gntotal_generators, gntotal_closed);
	OutputOneGenerator(gndb_size);
	OutputOneClosedPat(gndb_size);

	if(gntotal_freqitems==0)
	{
		gnmax_pattern_len = 0;
		delete gpdata;
		DelIntArray(pitem_sup_map, ncapacity);
	}
	else if(gntotal_freqitems==1)
	{
		WriteOnePair(gntotal_generators, gntotal_closed);
		OutputOneGenerator(nfreq_item, pitem_sup_map[nfreq_item]);
		gpfull_itemset[gnfull_len++] = nfreq_item;
		OutputOneClosedPat(pitem_sup_map[nfreq_item]);
		gnfull_len--;

		gnmax_pattern_len = 1;
		delete gpdata;
		DelIntArray(pitem_sup_map, ncapacity);
	}
	else if(gntotal_freqitems>1)
	{
		pheader_table = NewHeaderTable(gntotal_freqitems);
		gntotal_freqitems = 0;
		for(i=0;i<gnmax_item_id;i++)
		{
			if(pitem_sup_map[i]>=gnmin_sup && pitem_sup_map[i]<gndb_size)
			{
				pheader_table[gntotal_freqitems].nitem = i;
				pheader_table[gntotal_freqitems].nsupport = pitem_sup_map[i];
				pheader_table[gntotal_freqitems].pconddb = NULL;
				gntotal_freqitems++;
			}
		}
		qsort(pheader_table, gntotal_freqitems, sizeof(HEADER_NODE), comp_hnode_freq_des);
		DelIntArray(pitem_sup_map, ncapacity);

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
		goFPtree.DepthFGGrowth(proot, pheader_table, gntotal_freqitems, 0);
		DelIntArray(gpdfs_item_suporder_map, gntotal_freqitems);

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

	//PrintSummary();

}


void FPtree::DepthFGGrowth(FP_NODE *proot, HEADER_TABLE pheader_table, int num_of_freqitems, int num_of_tail_items)
{
	HEADER_TABLE pnewheader_table;
	FP_NODE *pnewroot, *pfpnode;
	FPNODE_PAGE *pstart_page;
	int k, i, nstart_pos, ncount, nlen, nitem, nkey_pat_id, nclosed_pat_id;
	int num_of_tail_comm, num_of_comm_items, num_of_newfreqitems, num_of_newtail_items, num_of_newtail_comm;
	unsigned int norig_closed_mapvalue;
	int norig_full_len;
	bool bnonkey;

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

//if(gnfull_len>=7 && gpfull_itemset[1]==90 && gpfull_itemset[2]==23 && gpfull_itemset[3]==34 
//   && gpfull_itemset[4]==86 && gpfull_itemset[5]==93 && gpfull_itemset[6]==56)
//printf("stop\n");

		nkey_pat_id = gopatternset.InsertKey(pheader_table[k].nsupport);

		if(pheader_table[k].pconddb==NULL)
		{
			nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport);
			WriteOnePair(gntotal_generators-1, nclosed_pat_id);
		}
		else //if(pheader_table[k].pconddb!=NULL)
		{
			if(pheader_table[k].pconddb->pnode_link==NULL)
			{
				ncount = pheader_table[k].pconddb->frequency;
				pfpnode = pheader_table[k].pconddb->pparent;
				num_of_comm_items = 0;
				if(ncount==pheader_table[k].nsupport)
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
					nclosed_pat_id = goclosedset.InsertClosed(ncount);
					WriteOnePair(nkey_pat_id, nclosed_pat_id);
					for(i=0;i<num_of_comm_items+num_of_tail_comm;i++)
					{
						RemoveOneCommItem();
					}
				}
				else if(ncount>=gnmin_sup)
				{
					nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport);
					WriteOnePair(nkey_pat_id, nclosed_pat_id);

					nlen = 0;
					while(pfpnode!=NULL)
					{
						if(gnprefix_len==1)
							gpsingle_branch[nlen++] = pheader_table[pfpnode->nitem_order].nitem;
						else if(ncount!=pheader_table[pfpnode->nitem_order].nsupport)
						{
							if(IsGenerator(pheader_table[pfpnode->nitem_order].nitem, ncount))
								gpsingle_branch[nlen++] = pheader_table[pfpnode->nitem_order].nitem;
						}

						nitem = pheader_table[pfpnode->nitem_order].nitem;
						AddOneCommItem(nitem);
						num_of_comm_items++;
						pfpnode = pfpnode->pparent;
					}
					if(nlen>=1)
					{
						num_of_tail_comm = CountTailFullItems(pheader_table[k].pconddb, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, k);
						for(i=0;i<num_of_tail_comm;i++)
						{
							nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
							AddOneCommItem(nitem);
						}
						nclosed_pat_id = goclosedset.InsertClosed(ncount);
						for(i=0;i<nlen;i++)
						{
							nkey_pat_id = gopatternset.InsertKey(gpsingle_branch[i], ncount);
							WriteOnePair(nkey_pat_id, nclosed_pat_id);
						}
						for(i=0;i<num_of_tail_comm;i++)
						{
							RemoveOneCommItem();
						}
					}
					for(i=0;i<num_of_comm_items;i++)
					{
						RemoveOneCommItem();
					}
				}
				else
				{
					nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport);
					WriteOnePair(nkey_pat_id, nclosed_pat_id);
				}
			}
			else
			{
				//count frequent items from AFOPT-tree
				memset(gpdfs_item_suporder_map, 0, sizeof(int)*k);
				memset(&(gpdfs_item_suporder_map[k]), -1, sizeof(int)*(num_of_freqitems+num_of_tail_items-k));
				CountFreqItems(pheader_table, k, gpdfs_item_suporder_map);

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
						if(gpdfs_item_suporder_map[i]!=pheader_table[i].nsupport && (gnprefix_len==1 || IsGenerator(pheader_table[i].nitem, gpdfs_item_suporder_map[i])))
							gpdfs_item_counter[num_of_newfreqitems].nsupport = gpdfs_item_suporder_map[i];
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

				nclosed_pat_id = goclosedset.InsertClosed(pheader_table[k].nsupport);
				WriteOnePair(nkey_pat_id, nclosed_pat_id);

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
					nkey_pat_id = gopatternset.InsertKey(gpdfs_item_counter[0].nitem, gpdfs_item_counter[0].nsupport);

					AddOneCommItem(gpdfs_item_counter[0].nitem);

					num_of_newtail_comm = GetTailFullItems(pheader_table[k].pconddb, gpdfs_item_counter[0].nsupport, &(gpfull_itemset[gnfull_len]), num_of_freqitems+num_of_tail_items, gpdfs_item_counter[0].norder, gpdfs_item_suporder_map, bnonkey);
					for(i=0;i<num_of_newtail_comm;i++)
					{
						nitem = pheader_table[gpfull_itemset[gnfull_len]].nitem;
						AddOneCommItem(nitem);
					}
					nclosed_pat_id = goclosedset.InsertClosed(gpdfs_item_counter[0].nsupport);
					WriteOnePair(nkey_pat_id, nclosed_pat_id);

					for(i=0;i<num_of_newtail_comm;i++)
					{
						RemoveOneCommItem();
					}
					RemoveOneCommItem();
				}
				else if(num_of_newfreqitems>1)
				{
					pnewheader_table = NewHeaderTable(num_of_newfreqitems+num_of_newtail_items);
					for(i=0;i<num_of_newfreqitems+num_of_newtail_items;i++)
					{
						pnewheader_table[i].nitem = gpdfs_item_counter[i].nitem;
						pnewheader_table[i].nsupport = gpdfs_item_counter[i].nsupport;
						pnewheader_table[i].pconddb = NULL;
					}

					gnum_of_newfreqitems = num_of_newfreqitems+num_of_newtail_items;
					pstart_page = gofpnode_buf.pcur_page;
					nstart_pos = gofpnode_buf.ncur_pos;
					pnewroot = BuildNewTreeWTail(pheader_table[k].pconddb, pnewheader_table, gpdfs_item_suporder_map); 

					DepthFGGrowth(pnewroot, pnewheader_table, num_of_newfreqitems, num_of_newtail_items);

					Reset(pstart_page, nstart_pos);
					DelHeaderTable(pnewheader_table, num_of_newfreqitems+num_of_newtail_items);
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
	if(gdused_mem_size!=0)
		printf("Error with memory: %.2f are not released\n", gdused_mem_size);

	FILE *fp_sum;

	fp_sum = fopen("fg.sum.txt", "a+");
	if(fp_sum == NULL)
	{
		printf("Error[PrintSummary]: cannot open file fg.sum.txt\n");
		return;
	}
	fprintf(fp_sum, "FGGrowth-FBd %s ", goparameters.szdata_filename);
	fprintf(fp_sum, "%f ", (double)goparameters.nmin_sup*100/gndb_size);
	fprintf(fp_sum, "%d %d %d %d\t", gnmax_pattern_len, gntotal_generators, gntotal_closed, gngenerator_check_times);
	fprintf(fp_sum, "%.2f\t", gdtotal_running_time);
	fprintf(fp_sum, "%.2fMB %.2fMB %.2fMB\t", gdmax_used_mem_size/(1<<20), (double)gntree_init_size/(1<<20), (double)gntree_max_size/(1<<20));

	fprintf(fp_sum, "\n");
	fclose(fp_sum);

}

