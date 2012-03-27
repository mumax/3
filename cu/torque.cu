#include "geom.h"
#include "magnetization.h"
#include "alpha.h"

extern "C" {

__global__ void torque() {

	int i = threadindex;
	if (i >= Ncell) { return; }

	float mx, my, mz;
	float alpha;

	load_magnetization(mx, my, mz, i);
	load_alpha(alpha, i);
	
}


}
