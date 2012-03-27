#include "geom.h"
#include "magnetization.h"

extern "C" {

__global__ void torque() {

	int i = threadindex;

	float mx, my, mz;

	if (i < N) {
		load_magnetization(mx, my, mz, i);
	}
}

}

