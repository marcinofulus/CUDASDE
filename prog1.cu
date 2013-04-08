/*
 * The Stewart-McCumber model of phase dynamics in a Josephson junction.
 *
 * (C) 2009 Michal Januszewski, Marcin Kostur
 *     Institute of Physics, University of Silesia, Katowice
 *
 * This file is subject to the terms and conditions of the GNU General Public
 * License v3.
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <getopt.h>
#include <unistd.h>
#include <time.h>

#include "rng.cu"

/* Simulation parameters */
#define TRANSIENT_PERIODS 200
#define MAX_PERIODS 2000
__constant__ float force = 0.1f;
__constant__ float gam = 0.9f;
__constant__ float d0 = 0.001f;
__constant__ float omega = 4.9f;

/* Accuracy and performance-related parameters */
__constant__ float dt = 0.0f;
__constant__ int samples = 100;

int spp = 100;		// steps per period
int paths = 2048;	// number of paths to sample, must be a multiple of 64
float par_amp = 4.2f;

bool output_path = false;
bool output_histogram = false;
bool output_allpaths = false;
bool output_avgv = false;

__device__ inline void diffEq(float &nx, float &nv, float x, float v, float t, float lomega, float lamp)
{
	nx = v;
	nv = -2.0f * PI * cosf(2.0f * PI * x) + lamp * cosf(lomega * t) + force - gam * v;
}

__global__ void advanceSystem(unsigned int *rng_state, float *cx, float *cv, float *amp, float ct)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float n1, n2, x, v, t;
	float xim, vim, xt1, vt1, xt2, vt2;
	float lamp;

	unsigned int lrng_state;

	x = cx[idx];
	v = cv[idx];
	lamp = amp[idx];

	t = ct;

	lrng_state = rng_state[idx];

	for (i = 1; i <= samples; i++) {
		n1 = rng_uni(&lrng_state);
		n2 = rng_uni(&lrng_state);

		bm_trans(n1, n2);

		diffEq(xt1, vt1, x, v, t, omega, lamp);

		xim = x + xt1 * dt;
		vim = v + vt1 * dt + sqrtf(dt * gam * d0 * 2.0f) * n1;
		t = ct + i * dt;

		diffEq(xt2, vt2, xim, vim, t, omega, lamp);

		x += 0.5f * dt * (xt1 + xt2);
		v += 0.5f * dt * (vt1 + vt2) + sqrtf(2.0f * dt * gam * d0) * n2;
	}

	cx[idx] = x;
	cv[idx] = v;

	rng_state[idx] = lrng_state;;
}

static struct option options[] = {
	{ "amp", required_argument, NULL, 'a' },
	{ "force", required_argument, NULL, 'f' },
	{ "gamma", required_argument, NULL, 'g' },
	{ "noise", required_argument, NULL, 'd' },
	{ "omega", required_argument, NULL, 'w' },
	{ "steps_per_period", required_argument, NULL, 's' },
	{ "paths", required_argument, NULL, 'p' },
	{ "mode", required_argument, NULL, 0x101 },
};

void usage(char **argv)
{
	printf("Usage: %s <params> [options]\n\n", argv[0]);
	printf("Required parameters:\n");
	printf("  --mode=MODE         Sets the output mode.  MODE can be one of: \n");
	printf("                        avgv, path, allpaths, hist\n");
	printf("                      allpaths: outputs the position for all paths\n");
	printf("                      avgv: <<v>> for multiple system parameters\n");
	printf("                      hist: the final position of all paths\n");
	printf("                      path: <v>(t)\n\n");
	printf("Other options:\n");
	printf("  -a, --amp=NUM       set the system parameter 'a' to NUM\n");
	printf("  -f, --force=NUM     set the system parameter 'f' to NUM\n");
	printf("  -g, --gamma=NUM     set the system parameter '\\gamma' to NUM\n");
	printf("  -w, --omega=NUM     set the system parameter '\\omega' to NUM\n");
	printf("  -s, --stps_per_period=NUM\n");
	printf("                      specify how many integration steps should be \n");
	printf("                      calculated for a single period of the driving force\n");
	printf("  -p, --paths=NUM     set the number of paths to NUM\n");
	printf("  -d, --noise=NUM     set the noise strength\n");

	printf("\nEXAMPLE using gnuplot and gsl-histogram: \n");
	printf("p '< ./prog1 --noise=0.0001 --amp=4.2  --mode=hist | gsl-histogram -150 100 50' u 2:3 w boxes \n");
	printf("p '< ./prog1 --noise=0.001  --amp=4.2  --mode=hist | gsl-histogram -150 100 50' u 2:3 w boxes \n");
  printf("p '< ./prog1 --noise=0.01   --amp=4.2  --mode=hist | gsl-histogram -150 100 50' u 2:3 w boxes \n");
	printf("  \n");

}

void parse_params(int argc, char **argv)
{
	int c;
	float tmp;

	while ((c = getopt_long(argc, argv, "a:f:g:d:w:s:p:", options, NULL)) != EOF) {
		switch (c) {

		case 'a':
			par_amp = atof(optarg);
			break;

		case 'f':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(force, &tmp, sizeof(float));
			break;

		case 'g':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(gam, &tmp, sizeof(float));
			break;

		case 'd':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(d0, &tmp, sizeof(float));
			break;

		case 'w':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(omega, &tmp, sizeof(float));
			break;

		case 's':
			spp = atoi(optarg);
			break;

		case 'p':
			paths = (atoi(optarg) / 64) * 64;
			break;

		case 0x101:
			if (!strcmp(optarg, "path"))
				output_path = true;
			else if (!strcmp(optarg, "hist"))
				output_histogram = true;
			else if (!strcmp(optarg, "allpaths"))
				output_allpaths = true;
			else if (!strcmp(optarg, "avgv"))
				output_avgv = true;
			break;
		}
	}
}

int main(int argc, char **argv)
{
	parse_params(argc, argv);

	if (!output_path && !output_allpaths && !output_histogram && !output_avgv) {
		usage(argv);
		return -1;
	}

	float par_omega;
	float par_gamma;

	cudaMemcpyFromSymbol(&par_omega, omega, sizeof(float));
	cudaMemcpyFromSymbol(&par_gamma, gam, sizeof(float));

	// Set the step size.
	float tmp = 2.0f * PI / par_omega / spp;

	cudaMemcpyToSymbol(dt, &tmp, sizeof(float));
	cudaMemcpyToSymbol(samples, &spp, sizeof(int));

	int num_threads = paths;

	if (output_avgv) {
		num_threads *= 100;
	}

	float *x, *v, *xbegin, *amp;
	float *dx, *dv, *damp;
	unsigned int *rng_state, *drng_state;

	size_t size = num_threads * sizeof(float);
	size_t size2 = num_threads * sizeof(unsigned int);
	int i, j;
	long step;

	xbegin = (float*)malloc(size);
	x = (float*)malloc(size);
	v = (float*)malloc(size);
	amp = (float*)malloc(size);
	rng_state = (unsigned int*)malloc(size2);
	tmp = par_amp;

	for (j = 0; j < num_threads / paths; j++) {
		for (i = 0; i < paths; i++) {
			amp[j * paths + i] = tmp;
		}

		tmp += 0.1f;
	}

	for (i = 0; i < num_threads; i++) {
		x[i] = 0.0f;
		v[i] = 0.0f;
	}

	srandom(time(0));

	for (i = 0; i < num_threads; i++) {
		rng_state[i] = (unsigned int)random();
	}

	cudaMalloc((void**)&drng_state, size2);
	cudaMalloc((void**)&dx, size);
	cudaMalloc((void**)&dv, size);
	cudaMalloc((void**)&damp, size);

	cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dv, v, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drng_state, rng_state, size2, cudaMemcpyHostToDevice);
	cudaMemcpy(damp, amp, size, cudaMemcpyHostToDevice);

	for (step = 0; step < (long)(spp * MAX_PERIODS); step += spp) {
		float t = step * 2.0f * PI / par_omega / (float)spp;

		advanceSystem<<<num_threads/64, 64>>>(drng_state, dx, dv, damp, t);

		// Ensemble-averaged x(t).
		if (output_path) {
			cudaMemcpy(x, dx, size, cudaMemcpyDeviceToHost);
			float sx = 0.0f;
			for (i = 0; i < paths; i++) {
				sx += x[i];
			}
			printf("%e %e\n", t, sx/paths);

		// Individual paths (x_i(t))
		} else if (output_allpaths) {
			cudaMemcpy(x, dx, size, cudaMemcpyDeviceToHost);
			for (i = 0; i < paths; i++) {
				printf("%e %d %e\n", t, i, x[i]);
			}
			printf("\n");

		// Save particle positions for evaluation of the asymptotic velocity <<v>>.
		} else if (output_avgv) {
			if ( step == (TRANSIENT_PERIODS*spp) ) {
				cudaMemcpy(xbegin, dx, size, cudaMemcpyDeviceToHost);
			}
		}
	}

	// Asymptotic velocity <<v>>.
	if (output_avgv) {
		cudaMemcpy(x, dx, size, cudaMemcpyDeviceToHost);

		for (j = 0; j < num_threads/paths; j++) {
			float sx1 = 0.0f;
			float sx2 = 0.0f;

			for (i = 0; i < paths; i++) {
				sx1 += xbegin[j*paths + i];
				sx2 += x[j*paths + i];
			}

			printf("%e %e %e %e\n", (sx2 - sx1) / paths / ((MAX_PERIODS - TRANSIENT_PERIODS) * 2.0f * PI / par_omega),
									amp[j*paths], par_omega, par_gamma);
		}

	// Individual particle positions at the end of simulation (x_i(t=t_final)).
	} else if (output_histogram) {
		cudaMemcpy(x, dx, size, cudaMemcpyDeviceToHost);

		for (i = 0; i < paths; i++) {
			printf("%e\n", x[i]);
		}
	}

	free(amp);
	free(xbegin);
	free(x);
	free(v);
	free(rng_state);

	cudaFree(damp);
	cudaFree(drng_state);
	cudaFree(dx);
	cudaFree(dv);

	return 0;
}

