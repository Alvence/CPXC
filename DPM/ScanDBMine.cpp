#include <stdlib.h>
#include <stdio.h>

#include "ScanDBMine.h"
#include "data.h"
#include "inline_routine.h"

CScanDBMine  goDBMiner;

// Enumerate all the frequent items in the database
int CScanDBMine::ScanDBCountFreqItems(int* &pitem_classsup_map)
{
	Transaction* ptransaction;
	int *ptemp_map, ncapacity, nnewcapacity;
	int i, nclass_id;

	ncapacity = ITEM_SUP_MAP_SIZE;
	pitem_classsup_map = NewIntArray(ncapacity*gnum_of_classes, 0);

	gndb_size = 0;
	gnmax_trans_len = 0;
	gnmax_item_id = 0;

	ptransaction=gpdata->getNextTransaction();
	while(ptransaction)
	{
		if(ptransaction->length>0)
		{
			gndb_size++;
			if(gnmax_trans_len<ptransaction->length)
				gnmax_trans_len = ptransaction->length;
			nclass_id = ptransaction->t[0];
			for(i=0;i<ptransaction->length;i++)
			{
				if(gnmax_item_id<ptransaction->t[i])
					gnmax_item_id = ptransaction->t[i];

				if(ptransaction->t[i]>=ncapacity)
				{
					nnewcapacity = MAX(2*ncapacity, ptransaction->t[i]+1);
					ptemp_map = NewIntArray(nnewcapacity*gnum_of_classes);
					memcpy(ptemp_map, pitem_classsup_map, sizeof(int)*ncapacity*gnum_of_classes);
					memset(&(ptemp_map[ncapacity]), 0, sizeof(int)*(nnewcapacity-ncapacity)*gnum_of_classes);
					DelIntArray(pitem_classsup_map, ncapacity*gnum_of_classes);
					pitem_classsup_map = ptemp_map;
					ncapacity = nnewcapacity;
				}
				pitem_classsup_map[ptransaction->t[i]*gnum_of_classes+nclass_id]++;
			}
		}
		ptransaction = gpdata->getNextTransaction();
	}

	gnmax_trans_len++;
	gnmax_item_id++;

	return ncapacity;
}


FP_NODE* CScanDBMine::ScanDBBuildFPtree(HEADER_TABLE pheader_table, int *pitem_order_map)
{
	Transaction *ptransaction;
	int ntrans_len, i, order, nclass_id;
	FP_NODE *proot;

	proot = NULL;
	ptransaction = gpdata->getNextTransaction();
	while(ptransaction)
	{
		if(ptransaction->length>0)
		{
			nclass_id = ptransaction->t[0];
			ntrans_len = 0;
			for(i=1;i<ptransaction->length;i++)
			{
				order = pitem_order_map[ptransaction->t[i]];
				if(order>=0)
				{
					gptransaction[ntrans_len] = order;
					ntrans_len++;
				}
			}

			if(ntrans_len>1)
			{
				sort_trans(gptransaction, ntrans_len);
				goFPtree.InsertTransaction(proot, pheader_table, gptransaction, ntrans_len, 1, nclass_id);
			}
		}
		ptransaction = gpdata->getNextTransaction();
	}

	return proot;
}

