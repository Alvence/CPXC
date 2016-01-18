#include <stdlib.h>

#include "PatternSet.h"
#include "Global.h"
#include "inline_routine.h"
#include "fsout.h"

PatternSet gopatternset;
int *gpitem_order_map;
unsigned int gnmap_value;
unsigned short *gpprefix_orderset;

PatternSet goclosedset;
unsigned int gnclosed_mapvalue;
unsigned short *gpfull_orderset;


int gnmax_search_depth;
double gdavg_search_depth;
int gnsearch_times;


void PatternSet::Init()
{
	mopatternset.phead = NewPatPage();
	mopatternset.ptail = mopatternset.phead;
	mopatternset.ncur_pos = 0;

	mppat_map = NewPatMap(0);
	mppat_map->pprev = NULL;

	momap_node_buf.phead = NewMapNodePage();
	momap_node_buf.ptail = momap_node_buf.phead;
	momap_node_buf.ncur_pos = 0;

	gnmax_search_depth = 0;
	gdavg_search_depth = 0;
	gnsearch_times = 0;
}

int PatternSet::InsertKey(int nitem, int nsupport, int nminor_sum)
{
	int nkey_pat_id;

	gpprefix_itemset[gnprefix_len] = nitem;
	gpprefix_orderset[gnprefix_len] = gpitem_order_map[nitem];
	gnprefix_len++;
	gnmap_value = HashAdd1Item(gnmap_value, nitem);

	nkey_pat_id = InsertKey(nsupport, nminor_sum);
	
	gnmap_value = HashRemove1Item(gnmap_value, nitem);
	gnprefix_len--;

	return nkey_pat_id;
}

int PatternSet::InsertKey(int nsupport, int nminor_sum)
{

	unsigned int nvalue; 
	PAT_MAP_NODE* pmap_node;
	int nkey_pat_id;

	if(nminor_sum<=gndelta)
	{
		nkey_pat_id = gntotal_generators;
		OutputOneGenerator(nsupport);
		return nkey_pat_id; 
	}
	else
	{
		if(gnprefix_len>1)
		{
			nvalue = HashFunc(gnmap_value);

			//inserting a map node
			if(momap_node_buf.ncur_pos==MAP_NODE_PAGE_SIZE)
			{
				MAP_NODE_PAGE* pmap_node_page;

				pmap_node_page = NewMapNodePage();
				pmap_node_page->pnext = NULL;
				momap_node_buf.ptail->pnext = pmap_node_page;
				momap_node_buf.ptail = pmap_node_page;
				momap_node_buf.ncur_pos = 0;
			}
			pmap_node = &(momap_node_buf.ptail->pmapnodes[momap_node_buf.ncur_pos]);
			momap_node_buf.ncur_pos++;
			pmap_node->pnext = mppat_map->ppat_map_nodes[nvalue];
			mppat_map->ppat_map_nodes[nvalue] = pmap_node;
			mppat_map->num_of_nodes++;

			//inserting the pattern 
			if(mopatternset.ncur_pos+gnprefix_len+5>=PAT_PAGE_SIZE)
			{
				PAT_PAGE *ppat_page;

				//memset(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), -1, sizeof(unsigned short)*(PAT_PAGE_SIZE-mopatternset.ncur_pos));
				ppat_page = NewPatPage();
				ppat_page->pnext = NULL;
				mopatternset.ptail->pnext = ppat_page;
				mopatternset.ptail = ppat_page;
				mopatternset.ncur_pos = 0;
			}
			pmap_node->ppattern = &(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]);
			mopatternset.ptail->ppatterns[mopatternset.ncur_pos] = gnprefix_len;
			mopatternset.ncur_pos++;
			memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &nsupport, sizeof(int));
			mopatternset.ncur_pos += 2;
			memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), gpprefix_orderset, gnprefix_len*sizeof(unsigned short));
			mopatternset.ncur_pos += gnprefix_len;
			memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &gnmap_value, sizeof(unsigned int));
			mopatternset.ncur_pos += 2;
		}
		return -1;
	}
}

bool PatternSet::IsGenerator(int nitem, int nsupport)
{
	PAT_MAP_NODE *pmap_node;
	int length, i, j, nsubset_sup, nsubset_len;
	unsigned short *psubset;
	unsigned int nvalue, nsubset_map_value, nsubset_hash_value, npat_map_value;
	bool bisgenerator, bfound;
//	int nsearch_depth;

	nvalue = HashAdd1Item(gnmap_value, nitem);
	length = gnprefix_len+1;
	if(length<3)
		printf("Error with generator length\n");

	bisgenerator = true;
	for(i=length-3;i>=0;i--)
	{
		gpitem_bitmap[gpprefix_orderset[i]] = 0;

		nsubset_map_value = HashRemove1Item(nvalue, gpprefix_itemset[i]);
		nsubset_hash_value = HashFunc(nsubset_map_value);
		if(i==0)
		{
			PAT_MAP *ppat_map;
			int nmax_order;

			nmax_order = gpitem_order_map[nitem];
			ppat_map = mppat_map;
			if(ppat_map->nstart>nmax_order)
			{
				for(j=1;j<gnprefix_len && ppat_map->nstart>nmax_order;j++)
				{
					if(nmax_order<gpprefix_orderset[j])
						nmax_order = gpprefix_orderset[j];
				}
				while(ppat_map->nstart>nmax_order)
					ppat_map = ppat_map->pprev;
			}
			pmap_node = ppat_map->ppat_map_nodes[nsubset_hash_value];
		}
		else
			pmap_node = mppat_map->ppat_map_nodes[nsubset_hash_value];
		bfound = false;
		if(pmap_node!=NULL)
		{
//			nsearch_depth = 0;
			while(pmap_node!=NULL)
			{
//				nsearch_depth++;
				nsubset_len = pmap_node->ppattern[0];
				if(nsubset_len==gnprefix_len)
				{
					memcpy(&nsubset_sup, &(pmap_node->ppattern[1]), sizeof(int));
					if(nsubset_sup>=nsupport)
					{
						memcpy(&npat_map_value, &(pmap_node->ppattern[3+nsubset_len]), sizeof(unsigned int));
						if(npat_map_value==nsubset_map_value)
						{
							psubset = &(pmap_node->ppattern[3]);
							for(j=0;j<nsubset_len && gpitem_bitmap[psubset[j]];j++);
							if(j==nsubset_len)
							{
								bfound = true;
								if(nsubset_sup==nsupport)
									bisgenerator = false;
								break;
							}
						}
					}
				}
				pmap_node = pmap_node->pnext;
			}
//			if(gnmax_search_depth<nsearch_depth)
//				gnmax_search_depth=nsearch_depth;
//			gdavg_search_depth += nsearch_depth;
//			gnsearch_times++;
		}
		gpitem_bitmap[gpprefix_orderset[i]] = 1;
		if(!bfound)
			return false;
		if(!bisgenerator)
			return false;
	}
	return true;
}


void PatternSet::CheckMap(int nitem_order)
{
	if(mppat_map->num_of_nodes<MAX_MAP_NODE_NUM)
		return;

	PAT_MAP* ppat_map;
	ppat_map = NewPatMap(nitem_order);
	ppat_map->pprev = mppat_map;
	mppat_map = ppat_map;

}

void PatternSet::Destroy()
{
	MAP_NODE_PAGE* pmap_node_page;
	PAT_MAP *ppat_map;

//	GetMapStatis();

//	DelSubsetNodeArray(mpsubset_nodes, (1<<mnset_len));

	DelPatSet(&mopatternset);

	pmap_node_page = momap_node_buf.phead;
	while(pmap_node_page!=NULL)
	{
		momap_node_buf.phead = pmap_node_page->pnext;
		DelMapNodePage(pmap_node_page);
		pmap_node_page = momap_node_buf.phead;
	}

	ppat_map = mppat_map;
	while(ppat_map!=NULL)
	{
		mppat_map = ppat_map->pprev;
		DelPatMap(ppat_map);
		ppat_map = mppat_map;
	}
}


int PatternSet::InsertClosed(int nsupport, int* pclass_distr)
{
	PAT_MAP_NODE* pmap_node;
	int npat_id;
	unsigned int nhash_value;

	nhash_value = HashFunc(gnclosed_mapvalue);

	npat_id = FindClosed(nsupport);
	if(npat_id==-1)
	{
		npat_id = gntotal_closed;
		OutputOneClosedPat(nsupport, pclass_distr);

		//inserting a map node
		if(momap_node_buf.ncur_pos==MAP_NODE_PAGE_SIZE)
		{
			MAP_NODE_PAGE* pmap_node_page;

			pmap_node_page = NewMapNodePage();
			pmap_node_page->pnext = NULL;
			momap_node_buf.ptail->pnext = pmap_node_page;
			momap_node_buf.ptail = pmap_node_page;
			momap_node_buf.ncur_pos = 0;
		}
		pmap_node = &(momap_node_buf.ptail->pmapnodes[momap_node_buf.ncur_pos]);
		momap_node_buf.ncur_pos++;
		pmap_node->pnext = mppat_map->ppat_map_nodes[nhash_value];
		mppat_map->ppat_map_nodes[nhash_value] = pmap_node;
		mppat_map->num_of_nodes++;

		//inserting the pattern 
		if(mopatternset.ncur_pos+gnfull_len-gnglobal_full+7>=PAT_PAGE_SIZE)
		{
			PAT_PAGE *ppat_page;

			//memset(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), -1, sizeof(unsigned short)*(PAT_PAGE_SIZE-mopatternset.ncur_pos));
			ppat_page = NewPatPage();
			ppat_page->pnext = NULL;
			mopatternset.ptail->pnext = ppat_page;
			mopatternset.ptail = ppat_page;
			mopatternset.ncur_pos = 0;
		}
		pmap_node->ppattern = &(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]);
		mopatternset.ptail->ppatterns[mopatternset.ncur_pos] = gnfull_len-gnglobal_full;
		mopatternset.ncur_pos++;
		memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &nsupport, sizeof(int));
		mopatternset.ncur_pos += 2;
		memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &(gpfull_orderset[gnglobal_full]), (gnfull_len-gnglobal_full)*sizeof(unsigned short));
		mopatternset.ncur_pos += gnfull_len-gnglobal_full;
		memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &gnclosed_mapvalue, sizeof(unsigned int));
		mopatternset.ncur_pos += 2;
		memcpy(&(mopatternset.ptail->ppatterns[mopatternset.ncur_pos]), &npat_id, sizeof(unsigned int));
		mopatternset.ncur_pos += 2;
	}

	return npat_id;
}

int PatternSet::FindClosed(int nsupport)
{
//	return -1;

	PAT_MAP_NODE *pmap_node;
	int i, nset_sup, nset_len, npat_id;
	unsigned short *pitemset;
	unsigned int nhash_value, npat_map_value;
//	int nsearch_depth;

	npat_id = -1;

	nhash_value = HashFunc(gnclosed_mapvalue);

	pmap_node = mppat_map->ppat_map_nodes[nhash_value];
	if(pmap_node!=NULL)
	{
//		nsearch_depth = 0;
		while(pmap_node!=NULL)
		{
//			nsearch_depth++;
			nset_len = pmap_node->ppattern[0];
			if(nset_len==gnfull_len-gnglobal_full)
			{
				memcpy(&nset_sup, &(pmap_node->ppattern[1]), sizeof(int));
				if(nset_sup==nsupport)
				{
					memcpy(&npat_map_value, &(pmap_node->ppattern[3+nset_len]), sizeof(unsigned int));
					if(npat_map_value==gnclosed_mapvalue)
					{
						pitemset = &(pmap_node->ppattern[3]);
						for(i=0;i<nset_len && gpclosed_bitmap[pitemset[i]];i++);
						if(i==nset_len)
						{
							memcpy(&npat_id, &(pmap_node->ppattern[nset_len+5]), sizeof(unsigned int));
							break;
						}
					}
				}
			}
			pmap_node = pmap_node->pnext;
		}
//		if(gnmax_search_depth<nsearch_depth)
//			gnmax_search_depth=nsearch_depth;
//		gdavg_search_depth += nsearch_depth;
//		gnsearch_times++;
	}

	return npat_id;
}



void PatternSet::PrintMapStatis(FILE* fp_sum)
{
	gdavg_search_depth /= gnsearch_times;

	fprintf(fp_sum, "%d  %d %.2f %d %d\t", MAP_SIZE, gnmax_search_depth, gdavg_search_depth, gnsearch_times, ITEM_HASH_BITMAP);
}

void PatternSet::GetMapStatis()
{
	FILE *fp_stat, *fp_long;
	PAT_MAP *ppat_map;
	PAT_MAP_NODE *pmap_node;
	int num_of_maps, nmax_map_size, nlocal_nonempty_entries, nmax_nonempty_entries, nlocal_max_map_depth, nmax_map_depth;
	double davg_map_size, davg_nonempty_entries, dlocal_avg_map_depth, davg_map_depth; 
	int i, ndepth;
	unsigned int nmap_value, nhash_value;
	int j, length, nsupport;
	unsigned short* pitemset;

	fp_stat = fopen("map.stat.txt", "a+");
	if(fp_stat==NULL)
	{
		printf("Error: cannot open file map.stat.txt for write\n");
		return;
	}
	fprintf(fp_stat, "%s\t%d %d\n", goparameters.szdata_filename, gnmin_sup, gntotal_generators);
	fprintf(fp_stat, "Map Size: %d\n", MAP_SIZE);
	fprintf(fp_stat, "\n");

	fp_long = fopen("map.long.txt", "wt");
	if(fp_long==NULL)
	{
		printf("Error: cannot open file map.long.txt for write\n");
		return;
	}

	num_of_maps = 0;
	davg_map_size = 0;
	nmax_map_size = 0;
	nmax_nonempty_entries = 0;
	davg_nonempty_entries = 0;
	nmax_map_depth = 0;
	davg_map_depth = 0;

	ppat_map = mppat_map;
	while(ppat_map!=NULL)
	{
		num_of_maps++;
		davg_map_size += ppat_map->num_of_nodes;
		if(nmax_map_size<ppat_map->num_of_nodes)
			nmax_map_size = ppat_map->num_of_nodes; 

		nlocal_nonempty_entries = 0;
		nlocal_max_map_depth = 0;
		dlocal_avg_map_depth = 0;
		for(i=0;i<MAP_SIZE;i++)
		{
			pmap_node = ppat_map->ppat_map_nodes[i];
			if(pmap_node!=NULL)
			{
				nlocal_nonempty_entries++;
				ndepth = 0;
				while(pmap_node!=NULL)
				{
					ndepth++;
					pmap_node = pmap_node->pnext;
				}

				if(ndepth>=10)
				{
					fprintf(fp_long, "%d\t%d\n", i, ndepth);
					pmap_node = ppat_map->ppat_map_nodes[i];
					while(pmap_node!=NULL)
					{
						length = pmap_node->ppattern[0];
						memcpy(&nsupport, &(pmap_node->ppattern[1]), sizeof(int));
						pitemset = &(pmap_node->ppattern[3]);
						nmap_value = HashItemset(pitemset, length);
						nhash_value = HashFunc(nmap_value);

						fprintf(fp_long, "%d %d\t%d %d\t", nmap_value, nhash_value, length, nsupport);
						for(j=0;j<length;j++)
							fprintf(fp_long, "%d ", pitemset[j]);
						fprintf(fp_long, "\n");
						pmap_node = pmap_node->pnext;
					}
					fprintf(fp_long, "\n\n");
				}

				dlocal_avg_map_depth += ndepth;
				if(nlocal_max_map_depth<ndepth)
					nlocal_max_map_depth = ndepth;
			}
		}
		if(nmax_nonempty_entries<nlocal_nonempty_entries)
			nmax_nonempty_entries = nlocal_nonempty_entries;
		davg_nonempty_entries += nlocal_nonempty_entries; 
		if(nmax_map_depth<nlocal_max_map_depth)
			nmax_map_depth = nlocal_max_map_depth;
		davg_map_depth += dlocal_avg_map_depth;

		fprintf(fp_stat, "%d\t%d\t%d\t\t%d (%.3f)\t\t%d\t%.2f\n", num_of_maps, ppat_map->nstart, ppat_map->num_of_nodes, nlocal_nonempty_entries, (double)nlocal_nonempty_entries/MAP_SIZE, nlocal_max_map_depth, dlocal_avg_map_depth/nlocal_nonempty_entries);

		ppat_map = ppat_map->pprev;
	}

	fprintf(fp_stat, "\n");
	fprintf(fp_stat, "%d\t%d\t%d\t\t%d (%.3f)\t\t%d\t%.2f\n", num_of_maps, gntotal_freqitems, nmax_map_size, nmax_nonempty_entries, (double)nmax_nonempty_entries/MAP_SIZE, nmax_map_depth, davg_map_depth/davg_nonempty_entries);
	fprintf(fp_stat, "\t\t%.2f\t%.2f (%.3f)\t", davg_map_size/num_of_maps, davg_nonempty_entries/num_of_maps, davg_nonempty_entries/(num_of_maps*MAP_SIZE));
	fprintf(fp_stat, "\n\n\n");

	fclose(fp_stat);
	fclose(fp_long);
}

