#include <stdlib.h>
#include <stdio.h>


#include "FPtree.h"
#include "inline_routine.h"


FPtree goFPtree;


void FPtree::InsertTransaction(FP_NODE* &proot, HEADER_TABLE pheader_table, int* ptransaction, int length, int frequency)
{
	FP_NODE *pcur_node, *pnew_node, *pparent, *pleftsib;
	int i, j;

	pcur_node = proot;
	pparent = NULL;
	pleftsib = NULL;
	for(i=0;i<length;i++)
	{
		while(pcur_node!=NULL && pcur_node->nitem_order<ptransaction[i])
		{
			pleftsib = pcur_node;
			pcur_node = pcur_node->prightsibling;
		}

		if(pcur_node==NULL || pcur_node->nitem_order>ptransaction[i])
		{
			pnew_node = NewOneFPNode();
			pnew_node->nitem_order = ptransaction[i];
			pnew_node->frequency = frequency;
			if(i<length-1)
				pnew_node->nchild_freq = frequency;
			else
				pnew_node->nchild_freq = 0;
			pnew_node->pchild = NULL;
			pnew_node->pparent = pparent;
			pnew_node->prightsibling = pcur_node;
			if(pleftsib!=NULL)
				pleftsib->prightsibling = pnew_node;
			else if(pparent!=NULL)
				pparent->pchild = pnew_node;
			pnew_node->pnode_link = pheader_table[pnew_node->nitem_order].pconddb;
			pheader_table[pnew_node->nitem_order].pconddb = pnew_node;
			if(i==0 && pleftsib==NULL)
				proot = pnew_node;
			pparent = pnew_node;
			for(j=i+1;j<length;j++)
			{
				pnew_node = NewOneFPNode();
				pnew_node->nitem_order = ptransaction[j];
				pnew_node->frequency = frequency;
				if(j<length-1)
					pnew_node->nchild_freq = frequency;
				else
					pnew_node->nchild_freq = 0;
				pnew_node->pchild = NULL;
				pnew_node->pparent = pparent;
				pparent->pchild = pnew_node;
				pnew_node->prightsibling = NULL;
				pnew_node->pnode_link = pheader_table[pnew_node->nitem_order].pconddb;
				pheader_table[pnew_node->nitem_order].pconddb = pnew_node;
				pparent = pnew_node;
			}
			break;
		}
		else 
		{
			pcur_node->frequency += frequency;
			if(i<length-1)
				pcur_node->nchild_freq += frequency;
			pparent = pcur_node;
			pcur_node = pcur_node->pchild;
			pleftsib = NULL;
		}
	}
}

void FPtree::CountFreqItems(HEADER_TABLE pheader_table, int nitem_order, int *pitem_sup_map)
{
	FP_NODE *pitemnode, *pfpnode;
	int ncount;

	pitemnode = pheader_table[nitem_order].pconddb;
	while(pitemnode!=NULL)
	{
		ncount = pitemnode->frequency;
		pfpnode = pitemnode->pparent;
		while(pfpnode!=NULL)
		{
			pitem_sup_map[pfpnode->nitem_order] += ncount;
			pfpnode = pfpnode->pparent;
		}
		pitemnode = pitemnode->pnode_link;
	}
}


FP_NODE* FPtree::BuildNewFPTree(FP_NODE *pconddb, HEADER_TABLE pnewheader_table, int *pitem_order_map)
{
	FP_NODE *pnewroot, *pitemnode, *pfpnode;
	int ncount, ntrans_len;

	pnewroot = NULL;

	pitemnode = pconddb;
	while(pitemnode!=NULL)
	{
		ncount = pitemnode->frequency;
		pfpnode = pitemnode->pparent;
		ntrans_len = 0;
		while(pfpnode!=NULL)
		{
			if(pitem_order_map[pfpnode->nitem_order]>=0)
			{
				gptransaction[ntrans_len] = pitem_order_map[pfpnode->nitem_order];
				ntrans_len++;
			}
			pfpnode = pfpnode->pparent;
		}
		if(ntrans_len>1)
		{
			sort_trans(gptransaction, ntrans_len);
			InsertTransaction(pnewroot, pnewheader_table, gptransaction, ntrans_len, ncount);
		}
		pitemnode = pitemnode->pnode_link;
	}

	return pnewroot;
}

FP_NODE* FPtree::BuildNewTreeWTail(FP_NODE *pconddb, HEADER_TABLE pnewheader_table, int *pitem_order_map)
{
	TAIL_NODE *ptail_node;
	FP_NODE *pnewroot, *pitemnode, *pfpnode;
	int ncount, ntrans_len, i;

	pnewroot = NULL;

	ptail_node = gptailnodes_head;
	pitemnode = pconddb;
	while(pitemnode!=NULL)
	{
		if(pitemnode!=ptail_node->plocal_root)
			printf("Error: inconsistent FP node\n");
		ncount = pitemnode->frequency;
		pfpnode = pitemnode->pparent;
		ntrans_len = 0;
		while(pfpnode!=NULL)
		{
			if(pitem_order_map[pfpnode->nitem_order]>=0)
				gptransaction[ntrans_len++] = pitem_order_map[pfpnode->nitem_order];
			pfpnode = pfpnode->pparent;
		}
		if(ntrans_len>0)
		{
			if(ntrans_len>1)
				sort_trans(gptransaction, ntrans_len);
			for(i=0;i<ptail_node->num_of_local_full;i++)
			{
				if(pitem_order_map[ptail_node->plocal_fullitems[i]]>=0)
					gptransaction[ntrans_len++] = pitem_order_map[ptail_node->plocal_fullitems[i]];
			}
			if(ntrans_len>1)
				InsertTransaction(pnewroot, pnewheader_table, gptransaction, ntrans_len, ncount);
		}
		pitemnode = pitemnode->pnode_link;
		ptail_node = ptail_node->pnext;
	}
	return pnewroot;
}

void FPtree::countsup(FP_NODE *pfpnode)
{
	FP_NODE *pchild_node;

	pchild_node = pfpnode->pchild;
	while(pchild_node!=NULL)
	{
		gplocal_item_sup_map[pchild_node->nitem_order] += pchild_node->frequency;
		if(pchild_node->pchild!=NULL && pchild_node->nchild_freq==pchild_node->frequency)
			countsup(pchild_node);
		pchild_node = pchild_node->prightsibling;
	}
}

int FPtree::CountTailFullItems(FP_NODE *proot, int *pfull_items, int num_of_local_items, int ncur_order)
{
	FP_NODE *pfpnode;
	int i, num_of_comm_items;

	memset(gplocal_item_sup_map, 0, sizeof(int)*num_of_local_items);

	if(proot->nchild_freq<proot->frequency)
		return 0;
	else if(proot->pchild!=0)
	{
		pfpnode = proot->pchild;
		while(pfpnode!=NULL)
		{
			gplocal_item_sup_map[pfpnode->nitem_order] += pfpnode->frequency;
			if(pfpnode->pchild!=NULL && pfpnode->nchild_freq==pfpnode->frequency)
				countsup(pfpnode);
			pfpnode = pfpnode->prightsibling;
		}
	}

	num_of_comm_items = 0;
	for(i=ncur_order+1;i<num_of_local_items;i++)
	{
		if(gplocal_item_sup_map[i]==proot->frequency)
			pfull_items[num_of_comm_items++] = i;
	}

	return num_of_comm_items;
}

int FPtree::CountTailFullItems(FP_NODE *pconddb, int nfull_sup, int *pfull_items, int num_of_local_items, int ncur_order, bool bnew_tail)
{
	FP_NODE *pfpnode, *pchild_node;
	TAIL_NODE *ptail_node=NULL, *pprev_tail_node;
	PAT_PAGE *pprev_page, *ppat_page;
	int i, j, num_of_comm_items;

	memset(gplocal_item_sumsup_map, 0, sizeof(int)*num_of_local_items);
	gptailnodes_tail = gptailnodes_head;
	pprev_tail_node = NULL;
	gplocal_fullitems.ptail = gplocal_fullitems.phead;
	gplocal_fullitems.ncur_pos = 0;
	pprev_page = NULL;


	pfpnode = pconddb;
	while(pfpnode!=NULL)
	{
		if(bnew_tail)
		{
			if(gptailnodes_tail!=NULL)
			{
				ptail_node = gptailnodes_tail;
				pprev_tail_node = gptailnodes_tail;
				gptailnodes_tail = gptailnodes_tail->pnext;
			}
			else 
			{
				ptail_node = NewTailNode();
				pprev_tail_node->pnext = ptail_node;
				pprev_tail_node = ptail_node;
			}
			ptail_node->plocal_root = pfpnode;
			ptail_node->num_of_local_full = 0;
			ptail_node->plocal_fullitems = NULL;
		}

		if(pfpnode->nchild_freq==pfpnode->frequency)
		{
			memset(gplocal_item_sup_map, 0, sizeof(int)*num_of_local_items);
			pchild_node = pfpnode->pchild;
			while(pchild_node!=NULL)
			{
				gplocal_item_sup_map[pchild_node->nitem_order] += pchild_node->frequency;
				if(pchild_node->pchild!=NULL && pchild_node->nchild_freq==pchild_node->frequency)
					countsup(pchild_node);
				pchild_node = pchild_node->prightsibling;
			}
			for(i=ncur_order+1;i<num_of_local_items;i++)
			{
				if(gplocal_item_sup_map[i]==pfpnode->frequency)
				{
					gplocal_item_sumsup_map[i] += gplocal_item_sup_map[i];
					if(bnew_tail)
						ptail_node->num_of_local_full++;
				}
			}
			if(bnew_tail && ptail_node->num_of_local_full>0)
			{
				if(gplocal_fullitems.ncur_pos+ptail_node->num_of_local_full>PAT_PAGE_SIZE)
				{
					if(gplocal_fullitems.ptail->pnext==NULL)
					{
						ppat_page = NewPatPage();
						gplocal_fullitems.ptail->pnext = ppat_page;
						gplocal_fullitems.ptail = ppat_page;

					}
					else
						gplocal_fullitems.ptail = gplocal_fullitems.ptail->pnext;
					gplocal_fullitems.ncur_pos = 0;
				}

				ptail_node->plocal_fullitems = &(gplocal_fullitems.ptail->ppatterns[gplocal_fullitems.ncur_pos]);
				gplocal_fullitems.ncur_pos += ptail_node->num_of_local_full;

				j = 0;
				for(i=ncur_order+1;i<num_of_local_items;i++)
				{
					if(gplocal_item_sup_map[i]==pfpnode->frequency)
						ptail_node->plocal_fullitems[j++] = i;
				}
			}
		}

		pfpnode = pfpnode->pnode_link;
	}

	num_of_comm_items = 0;
	for(i=ncur_order+1;i<num_of_local_items;i++)
	{
		if(gplocal_item_sumsup_map[i]==nfull_sup)
			pfull_items[num_of_comm_items++] = i;
	}
	return num_of_comm_items;
}


int FPtree::GetTailFullItems(FP_NODE *pconddb, int nfull_sup, int *pfull_items, int num_of_local_items, int ncur_order, int*pitem_order_map, bool bnonkey)
{
	TAIL_NODE *ptail_node;
	FP_NODE *pfpnode, *pitemnode;
	int num_of_comm_items, i, ntrans_len;
	bool bcontained, bcount_tail_intree, bcount_tailitems;

	memset(gplocal_item_sumsup_map, 0, sizeof(int)*num_of_local_items);
	num_of_comm_items = 0;

	bcount_tail_intree = bnonkey;
	bcount_tailitems = true;

	pitemnode = pconddb;
	ptail_node = gptailnodes_head;
	while(pitemnode!=NULL)
	{
		if(pitemnode!=ptail_node->plocal_root)
			printf("Error: inconsistent FP node\n");
		bcontained = false;
		ntrans_len = 0;
		pfpnode = pitemnode->pparent;
		if(bcount_tail_intree)
		{
			while(pfpnode!=NULL)
			{
				if(pfpnode->nitem_order==ncur_order)
					bcontained = true;
				else if(pitem_order_map[pfpnode->nitem_order]>=0)
					gptransaction[ntrans_len++] = pfpnode->nitem_order;
				pfpnode = pfpnode->pparent;			
			}
		}
		else
		{
			while(pfpnode!=NULL)
			{
				if(pfpnode->nitem_order==ncur_order)
				{
					bcontained = true;
					break;
				}
				pfpnode = pfpnode->pparent;			
			}
		}
		if(bcontained)
		{
			if(bcount_tail_intree)
			{
				if(ntrans_len>0)
				{
					for(i=0;i<ntrans_len;i++)
						gplocal_item_sumsup_map[gptransaction[i]] += pitemnode->frequency;
				}
				else
					bcount_tail_intree = false;

			}
			if(bcount_tailitems)
			{
				if(ptail_node->num_of_local_full>0)
				{
					for(i=0;i<ptail_node->num_of_local_full;i++)
					{
						if(gpdfs_item_suporder_map[ptail_node->plocal_fullitems[i]]>=0)
							gplocal_item_sumsup_map[ptail_node->plocal_fullitems[i]] += ptail_node->plocal_root->frequency;
					}
				}
				else 
					bcount_tailitems = false;
			}
			if(!bcount_tailitems && !bcount_tail_intree)
				return 0;
		}
		ptail_node = ptail_node->pnext;
		pitemnode = pitemnode->pnode_link;
	}
	for(i=0;i<num_of_local_items;i++)
	{
		if(gplocal_item_sumsup_map[i]==nfull_sup)
			pfull_items[num_of_comm_items++] = i;
	}

	return num_of_comm_items;
}


bool FPtree::IsSinglePath(FP_NODE *proot)
{
	FP_NODE *pfpnode;
	
	pfpnode = proot;
	while(pfpnode!=NULL && pfpnode->prightsibling==NULL)
		pfpnode = pfpnode->pchild;
	if(pfpnode==NULL)
		return true;
	else 
		return false;
}

//sort items in descending frequency order
int comp_item_freq_des(const void *e1, const void *e2)
{
	ITEM_COUNTER *p1, *p2;
	p1 = (ITEM_COUNTER *) e1;
	p2 = (ITEM_COUNTER *) e2;

	if ((p1->nsupport>p2->nsupport) )
		return -1;
	else if (p1->nsupport<p2->nsupport)
		return 1;
	else 
		return 0;
}


//sort items in descending frequency order
int comp_hnode_freq_des(const void *e1, const void *e2)
{
	HEADER_NODE *p1, *p2;
	p1 = (HEADER_NODE *) e1;
	p2 = (HEADER_NODE *) e2;

	if ((p1->nsupport>p2->nsupport) )
		return -1;
	else if (p1->nsupport<p2->nsupport)
		return 1;
	else 
		return 0;
}
