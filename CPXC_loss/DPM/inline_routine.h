#pragma once 

#include <string.h>
#include "Global.h"
#include "parameters.h"

/*
// for detecting memory leak
#include <afx.h>
#define new DEBUG_NEW
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
//*/


inline void IncMemSize(int size)
{
	gdused_mem_size += size;
	if(gdused_mem_size>gdmax_used_mem_size)
		gdmax_used_mem_size = gdused_mem_size;
}
inline void DecMemSize(int size)
{
	gdused_mem_size -= size;
}



//=============================================================================
inline int* NewIntArray(int length)
{
	IncMemSize(length*sizeof(int));
	return new int[length];
}
inline void DelIntArray(int* parray, int length)
{
	DecMemSize(length*sizeof(int));
	delete []parray;
}
inline int** NewIntPointerArray(int length)
{
	IncMemSize(length*sizeof(int*));
	return new int*[length];
}
inline void DelIntPointerArray(int** ppitemsets, int length)
{
	DecMemSize(length*sizeof(int*));
	delete []ppitemsets;
}
inline int* NewIntArray(int length, int value)
{
	int *parray;

	parray = NewIntArray(length);
	memset(parray, value, sizeof(int)*length);
	return parray;
}
inline char* NewCharArray(int length)
{
	IncMemSize(length*sizeof(char));
	return new char[length];
}
inline void DelCharArray(char* parray, int length)
{
	DecMemSize(length*sizeof(char));
	delete []parray;
}
inline unsigned short* NewUShortArray(int length)
{
	IncMemSize(length*sizeof(unsigned short));
	return new unsigned short[length];
}
inline void DelUShortArray(unsigned short* parray, int length)
{
	DecMemSize(length*sizeof(unsigned short));
	delete []parray;
}

//=============================================================================


//-----------------------------------------------------------------------------
#include "FPtree.h"

#define FPNODE_PAGE_SIZE (1<<10)

struct FPNODE_PAGE
{
	FP_NODE *pfpnodes;
	FPNODE_PAGE *pnext;
};
struct FPNODE_BUF
{
	FPNODE_PAGE *phead;
	FPNODE_PAGE *pcur_page; 
	int ncur_pos;
	int ntotal_pages;
};
extern FPNODE_BUF gofpnode_buf;

inline FPNODE_PAGE* NewFPNodePage()
{
	FPNODE_PAGE *pfppage;

	pfppage = new FPNODE_PAGE;
	IncMemSize(sizeof(FPNODE_PAGE));
	pfppage->pfpnodes = new FP_NODE[FPNODE_PAGE_SIZE];
	IncMemSize(sizeof(FP_NODE)*FPNODE_PAGE_SIZE);
	pfppage->pnext = NULL;

	return pfppage;
}
inline void DelFPNodePage(FPNODE_PAGE* pfppage)
{
	delete []pfppage->pfpnodes;
	DecMemSize(sizeof(FP_NODE)*FPNODE_PAGE_SIZE);
	delete pfppage;
	DecMemSize(sizeof(FPNODE_PAGE));
}

inline FP_NODE* NewOneFPNode()
{
	FP_NODE *pfpnode;
	FPNODE_PAGE *pnewfppage;

	if(gofpnode_buf.ncur_pos==FPNODE_PAGE_SIZE)
	{
		if(gofpnode_buf.pcur_page->pnext==NULL)
		{
			pnewfppage = NewFPNodePage();
			gofpnode_buf.pcur_page->pnext = pnewfppage;
			gofpnode_buf.pcur_page = pnewfppage;
			gofpnode_buf.ntotal_pages++;
		}
		else 
			gofpnode_buf.pcur_page = gofpnode_buf.pcur_page->pnext;
		gofpnode_buf.ncur_pos = 0;
	}
	
	pfpnode = &(gofpnode_buf.pcur_page->pfpnodes[gofpnode_buf.ncur_pos]);
	gofpnode_buf.ncur_pos++;
	
	return pfpnode;
}


struct FPCD_PAGE
{
	int *pclass_distrs;
	FPCD_PAGE *pnext;
};

struct FPCD_BUF
{
	FPCD_PAGE *phead;
	FPCD_PAGE *pcur_page;
	int ncur_pos;
	int ntotal_pages;
};
extern FPCD_BUF gofpcd_buf;
extern int gnfpcd_page_size;

inline FPCD_PAGE* NewFPCDNodePage()
{
	FPCD_PAGE *pcldis_page;

	pcldis_page = new FPCD_PAGE;
	IncMemSize(sizeof(FPCD_PAGE));
	pcldis_page->pclass_distrs = NewIntArray(gnfpcd_page_size);
	pcldis_page->pnext = NULL;

	return pcldis_page;
}
inline void DelFPCDNodePage(FPCD_PAGE* pcldis_page)
{
	DelIntArray(pcldis_page->pclass_distrs, gnfpcd_page_size);
	delete pcldis_page;
	DecMemSize(sizeof(FPCD_PAGE));
}

inline int* NewOneFPCDNode()
{
	FPCD_PAGE *pcldis_page;
	int *pclass_distr;

	if(gofpcd_buf.ncur_pos==gnfpcd_page_size)
	{
		if(gofpcd_buf.pcur_page->pnext==NULL)
		{
			pcldis_page = NewFPCDNodePage();
			gofpcd_buf.pcur_page->pnext = pcldis_page;
			gofpcd_buf.pcur_page = pcldis_page;
			gofpcd_buf.ntotal_pages++;
		}
		else 
			gofpcd_buf.pcur_page = gofpcd_buf.pcur_page->pnext;
		gofpcd_buf.ncur_pos = 0;
	}
	
	pclass_distr = &(gofpcd_buf.pcur_page->pclass_distrs[gofpcd_buf.ncur_pos]);
	gofpcd_buf.ncur_pos += gnum_of_classes;

	memset(pclass_distr, 0, sizeof(int)*gnum_of_classes);
	
	return pclass_distr;
}


//--------------------------------------------------------------------------



//=============================================================================
extern HEADER_TABLE gpdfs_header_array;
extern int gndfs_header_size;
extern int gndfs_header_pos;
extern int* gpdfs_classdistr_array;
extern int gndfs_classdistr_pos;
extern int gndfs_classdistr_size;

inline HEADER_TABLE NewHeaderTable(int num_of_freqitems)
{
	HEADER_TABLE pheader_table;

	if(gndfs_header_pos+num_of_freqitems<=gndfs_header_size)
	{
		pheader_table = &(gpdfs_header_array[gndfs_header_pos]);
		gndfs_header_pos += num_of_freqitems;
	}
	else 
	{
		IncMemSize(num_of_freqitems*sizeof(HEADER_NODE));
		pheader_table = new HEADER_NODE[num_of_freqitems];
	}

	return pheader_table;
}

inline void DelHeaderTable(HEADER_TABLE pheader_table, int num_of_freqitems)
{
	if(pheader_table>=gpdfs_header_array && pheader_table<gpdfs_header_array+sizeof(HEADER_NODE)*gndfs_header_size)
	{
		gndfs_header_pos -= num_of_freqitems;
	}
	else 
	{
		DecMemSize(num_of_freqitems*sizeof(HEADER_NODE));
		delete []pheader_table;
	}
}

inline int* NewDFSClassDistrs(int size)
{
	int *pclass_distrs;

	if(gndfs_classdistr_pos+size<=gndfs_classdistr_size)
	{
		pclass_distrs = &(gpdfs_classdistr_array[gndfs_classdistr_pos]);
		gndfs_classdistr_pos += size;
	}
	else
	{
		pclass_distrs = NewIntArray(size);
	}
	return pclass_distrs;
}
inline void DelDFSClassDistrs(int* pclass_distrs, int size)
{
	if(pclass_distrs>=gpdfs_classdistr_array && pclass_distrs<gpdfs_classdistr_array+sizeof(int)*gndfs_classdistr_size)
		gndfs_classdistr_pos -= size;
	else
		DelIntArray(pclass_distrs, size);
        ;
}

inline ITEM_COUNTER* NewItemCounter(int length)
{
	IncMemSize(sizeof(ITEM_COUNTER)*length);
	return new ITEM_COUNTER[length];
}
inline void DelItemCounter(ITEM_COUNTER* pcounters, int length)
{
	DecMemSize(sizeof(ITEM_COUNTER)*length);
	delete []pcounters;
}


//=============================================================================



//------------------------------------------------------------------------------

#include "PatternSet.h"

inline PAT_PAGE* NewPatPage()
{
	IncMemSize(sizeof(PAT_PAGE));

	PAT_PAGE *ppat_page;
	ppat_page = new PAT_PAGE;
	ppat_page->pnext = NULL;
	return ppat_page;
}
inline void DelPatPage(PAT_PAGE *ppat_page)
{
	DecMemSize(sizeof(PAT_PAGE));
	delete ppat_page;
}
inline void DelPatSet(PAT_SET *ppat_set)
{
	PAT_PAGE *ppat_page;

	ppat_page = ppat_set->phead;
	while(ppat_page!=NULL)
	{
		ppat_set->phead = ppat_page->pnext;
		DelPatPage(ppat_page);
		ppat_page = ppat_set->phead;
	}
}

inline PAT_MAP* NewPatMap(int nstart)
{
	IncMemSize(sizeof(PAT_MAP));

	PAT_MAP *ppat_map;
	ppat_map = new PAT_MAP;
	ppat_map->nstart = nstart;
	ppat_map->num_of_nodes = 0;
	for(int i=0;i<MAP_SIZE;i++)
		ppat_map->ppat_map_nodes[i] = NULL;

	return ppat_map;
}
inline void DelPatMap(PAT_MAP* ppat_map)
{
	DecMemSize(sizeof(PAT_MAP));
	delete ppat_map;
}

inline MAP_NODE_PAGE* NewMapNodePage()
{
	MAP_NODE_PAGE *pmapnode_page;

	IncMemSize(sizeof(MAP_NODE_PAGE));
	pmapnode_page = new MAP_NODE_PAGE;
	pmapnode_page->pnext = NULL;
	return pmapnode_page;
}
inline void DelMapNodePage(MAP_NODE_PAGE* pmapnode_page)
{
	DecMemSize(sizeof(MAP_NODE_PAGE));
	delete pmapnode_page;
}

//------------------------------------------------------------------------------


//=============================================================================
inline TAIL_NODE* NewTailNode()
{
	TAIL_NODE *pnode;

	gnum_of_tailnodes++;

	IncMemSize(sizeof(TAIL_NODE));
	pnode = new TAIL_NODE;
	pnode->pnext = NULL;
	return pnode;
}

inline void DelTailNodes(TAIL_NODE *phead)
{
	TAIL_NODE *pnode;

	pnode = phead;
	while(pnode!=NULL)
	{
		phead = pnode->pnext;
		DecMemSize(sizeof(TAIL_NODE));
		delete pnode;
		pnode = phead;
	}
}
//=============================================================================



//------------------------------------------------------------------------------
inline void Init()
{
	int length;

	length = MIN(gnmax_trans_len, gntotal_freqitems);
	gptransaction = NewIntArray(length);
	gpsort_trans_buf = NewCharArray(gntotal_freqitems);
	memset(gpsort_trans_buf, 0, sizeof(char)*gntotal_freqitems);
	gpsingle_branch = NewIntArray(length);

	//for dfs mining
	gndfs_header_size = gntotal_freqitems*3;
	gpdfs_header_array = new HEADER_NODE[gndfs_header_size];
	IncMemSize(sizeof(HEADER_NODE)*gndfs_header_size);
	gndfs_header_pos = 0;

	gndfs_classdistr_size = gndfs_header_size*gnum_of_classes;
	gpdfs_classdistr_array = NewIntArray(gndfs_classdistr_size);
	gndfs_classdistr_pos = 0;


    // for allocating space for fptree
	gofpnode_buf.phead = NewFPNodePage();
	gofpnode_buf.pcur_page = gofpnode_buf.phead;
	gofpnode_buf.ncur_pos = 0;
	gofpnode_buf.ntotal_pages = 1;

	gnfpcd_page_size = FPNODE_PAGE_SIZE*gnum_of_classes;
	gofpcd_buf.phead = NewFPCDNodePage();
	gofpcd_buf.pcur_page = gofpcd_buf.phead;
	gofpcd_buf.ncur_pos = 0;
	gofpcd_buf.ntotal_pages = 1;

}

inline void ResetFPBuf(FPNODE_PAGE *pcur_page, int ncur_pos)
{
	gofpnode_buf.pcur_page = pcur_page;
	gofpnode_buf.ncur_pos = ncur_pos;
}

inline void ResetFPCDBuf(FPCD_PAGE *pcur_page, int ncur_pos)
{
	gofpcd_buf.pcur_page = pcur_page;
	gofpcd_buf.ncur_pos = ncur_pos;

}


inline void Destroy()
{
	int length;

	length = MIN(gnmax_trans_len, gntotal_freqitems);
	DelIntArray(gptransaction, length);
	DelCharArray(gpsort_trans_buf, gntotal_freqitems);
	DelIntArray(gpsingle_branch, length);


	//for DFS mining
	DecMemSize(sizeof(HEADER_NODE)*gndfs_header_size);
	delete []gpdfs_header_array;
	DelIntArray(gpdfs_classdistr_array, gndfs_classdistr_size);

	//releasing fpnode buffer
	FPNODE_PAGE *pfppage, *pnext_fppage;
	pfppage = gofpnode_buf.phead;
	while(pfppage!=NULL)
	{
		pnext_fppage = pfppage->pnext;
		DelFPNodePage(pfppage);
		pfppage = pnext_fppage;	
		gofpnode_buf.ntotal_pages--;
	}
	if(gofpnode_buf.ntotal_pages!=0)
		printf("Error with number of FPNode pages\n");

	FPCD_PAGE *pfpcd_page, *pnext_fpcdpage;
	pfpcd_page = gofpcd_buf.phead;
	while(pfpcd_page!=NULL)
	{
		pnext_fpcdpage = pfpcd_page->pnext;
		DelFPCDNodePage(pfpcd_page);
		pfpcd_page = pnext_fpcdpage;
		gofpcd_buf.ntotal_pages--;
	}
	if(gofpcd_buf.ntotal_pages!=0)
		printf("Error with number of FPCDNode pages\n");
}

inline void AddOneCommItem(int nitem)
{
	if(nitem<0 || nitem>gnmax_item_id || gpitem_order_map[nitem]<0)
		printf("Error with item\n");
	if(gnfull_len>=gnmax_trans_len || gnfull_len>=gntotal_freqitems)
		printf("Error with full prefix length\n");


	gpfull_itemset[gnfull_len] = nitem;
	gpfull_orderset[gnfull_len] = gpitem_order_map[nitem];
	gpclosed_bitmap[gpitem_order_map[nitem]] = 1;
	gnfull_len++;
	gnclosed_mapvalue = HashAdd1Item(gnclosed_mapvalue, nitem);
}

inline void RemoveOneCommItem()
{

	gnfull_len--;
	gpclosed_bitmap[gpfull_orderset[gnfull_len]] = 0;
}
//------------------------------------------------------------------------------





inline void sort_trans(int *ptransaction, int ntrans_len)
{
	int i, j;

	if(gnum_of_newfreqitems/ntrans_len>64)
		qsort(ptransaction, ntrans_len, sizeof(int), comp_int_asc);
	else
	{
		for(i=0;i<ntrans_len;i++)
			gpsort_trans_buf[ptransaction[i]] = 1;

		j = 0;
		for(i=0;i<gnum_of_newfreqitems;i++)
		{
			if(gpsort_trans_buf[i])
			{
				ptransaction[j] = i;
				j++;
				gpsort_trans_buf[i] = 0;
			}
		}
	}

}
