#include <string.h>

#pragma warning (disable:4996)

#include "parameters.h"
#include "Global.h"

CParameters goparameters;


void CParameters::SetResultName(char *szresult_name)
{
	sprintf(szgenerator_filename, "%s.key", szresult_name);
	sprintf(szclosed_filename, "%s.closed", szresult_name);
	sprintf(szpair_filename, "%s.pair", szresult_name);
}

