#pragma once 


typedef struct FP_NODE
{
	int	frequency;
	int nchild_freq;
	int nitem_order;
	int *pclass_distr;
	FP_NODE *pchild;
	FP_NODE	*prightsibling;
	FP_NODE *pparent;
	FP_NODE *pnode_link;
} FP_NODE;

typedef struct
{
	int nitem; 
	int nsupport;
	int *pclass_distr;
	int nminor_sum; 
	FP_NODE *pconddb;
} HEADER_NODE;
typedef HEADER_NODE* HEADER_TABLE;


struct TAIL_NODE
{
	FP_NODE *plocal_root;
	int num_of_local_full;
	unsigned short *plocal_fullitems;
	TAIL_NODE *pnext;
};

extern TAIL_NODE *gptailnodes_head;
extern TAIL_NODE *gptailnodes_tail;
extern int gnum_of_tailnodes;
extern int gnum_of_tail_pages;
extern int *gplocal_item_sup_map;
extern int *gplocal_item_sumsup_map;


class FPtree
{
	void CountFreqItems(HEADER_TABLE pheader_table, int nitem_order, int *pitem_sup_map);
	FP_NODE* BuildNewFPTree(FP_NODE *pconddb, HEADER_TABLE pnewheader_table, int *pitem_order_map);

	void countsup(FP_NODE *pfpnode);
	int CountTailFullItems(FP_NODE *proot, int *pfull_items, int num_of_local_items, int ncur_order);
	int CountTailFullItems(FP_NODE *pconddb, int nfull_sup, int *pfull_items, int num_of_local_items, int ncur_order, bool bnewtail);

	int GetTailFullItems(FP_NODE *pconddb, int nfull_sup, int *pfull_items, int num_of_local_items, int ncur_order, int*pitem_order_map, bool bnonkey);

	FP_NODE* BuildNewTreeWTail(FP_NODE *pconddb, HEADER_TABLE pnewheader_table, int *pitem_order_map);

public:
	void InsertTransaction(FP_NODE* &proot, HEADER_TABLE pheader_table, int* ptransaction, int length, int frequency, int nclass_id);
	void InsertTransaction(FP_NODE* &proot, HEADER_TABLE pheader_table, int* ptransaction, int length, int ntail_pos, int frequency, int *pclass_distr);
	void DepthFGGrowth(FP_NODE *proot, HEADER_TABLE pheader_table, int num_of_freqitems, int num_of_tail_items);
	bool IsSinglePath(FP_NODE *proot);

};

extern FPtree goFPtree;
extern HEADER_TABLE gpheader_table;

int comp_item_freq_des(const void *e1, const void *e2);
int comp_hnode_freq_des(const void *e1, const void *e2);
