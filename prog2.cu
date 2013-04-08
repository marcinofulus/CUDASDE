/*
 * The general Kuramoto model.
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

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "rng.cu"

#define MAX_STEPS 2000
#define OUTPUT_NTH 100
#define BINS 64

__constant__ float T = 1.0f;
__constant__ float dt = 0.0f;
__constant__ float K = 4.0f;
__constant__ int num_particles = 0;

int sps = 100;		// steps per second
int niceness = 0;
int particles = 4096;

float hist_xmin = 0.0f;
float hist_xmax = 2.0f * PI;

bool output_histogram = false;
bool output_avg = false;

__global__ void updateRHS(float *x, float *v)
{
	int i, j, tile, idx, gidx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float shx[];

	float mx = x[gidx];
	float mv = 0.0f;

	for (i = 0, tile = 0; i < num_particles; i += blockDim.x, tile++) {
		idx = tile * blockDim.x + threadIdx.x;
		shx[threadIdx.x] = x[idx];

		__syncthreads();
		for (j = 0; j < blockDim.x; j++) {
			mv += sinf(shx[j] - mx);
		}
		__syncthreads();
	}

	v[gidx] = mv;
}

__global__ void advanceSystem(unsigned int *rng_state, float *x, float *v,float *f)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float lv = v[idx], lx = x[idx], lf = f[idx];
	float n1, n2;

	n1 = rng_uni(&rng_state[idx]);
	n2 = rng_uni(&rng_state[idx]);

	bm_trans(n1, n2);

	x[idx] = lx + (lf + K/num_particles * lv) * dt + sqrtf(2.0 * T * dt) * n1;
}

static struct option options[] = {
	{ "paths", required_argument, NULL, 'p' },
	{ "coupling", required_argument, NULL, 'K' },
	{ "noise", required_argument, NULL, 'T' },
	{ "mode", required_argument, NULL, 0x101 }
};

void parse_params(int argc, char **argv)
{
	int c;
	float tmp;
	while ((c = getopt_long(argc, argv, "p:K:T:", options, NULL)) != EOF) {
		switch (c) {
		case 'p':
			particles = (atoi(optarg) / 64) * 64;
			break;
		case 'T':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(T, &tmp, sizeof(float));
			break;
		case 'K':
			tmp = atof(optarg);
			cudaMemcpyToSymbol(K, &tmp, sizeof(float));
			break;
		case 0x101:
			if (!strcmp(optarg, "hist"))
				output_histogram = true;
			else if (!strcmp(optarg, "avg"))
				output_avg = true;
			break;
		}
	}
}

void usage(char **argv)
{
	printf("Usage: %s <params> [options]\n\n", argv[0]);
	printf("Required parameters:\n");
	printf("  --mode=MODE         MODE can be one of: hist, avg\n");
	printf("                      avg: outputs the sum of all sin(t) and cos(t) terms\n");
	printf("                      hist: outputs a position histogram\n\n");
	printf("Other options:\n");
	printf("  -p, --paths=NUM     set the number of paths to NUM\n");
	printf("  -K, --coupling=NUM  set the coupling constant K\n");
	printf("  -T, --noise=NUM     set the noise strength\n");
	printf("\nEXAMPLE using gnuplot: \n");
	printf("sp '< ./prog2  --paths=10000 --mode=hist' u 1:2:3 w l  \n");
}

int main(int argc, char **argv)
{
	int i, num_threads;

	float *f,*x, *v;
	float *df,*dx, *dv;

	gsl_rng_env_setup();

	parse_params(argc, argv);

	if (!output_histogram && !output_avg) {
		usage(argv);
		return -1;
	}

	num_threads = particles;
	cudaMemcpyToSymbol(num_particles, &particles, sizeof(int));

	unsigned int *rng_state, *drng_state;
	size_t size = num_threads * sizeof(float);
	size_t size2 = num_threads * sizeof(unsigned int);

	f = (float*)malloc(size);
	x = (float*)malloc(size);
	v = (float*)malloc(size);
	rng_state = (unsigned int*)malloc(size2);
	srand(time(0));

	for (i = 0; i < num_threads; i++) {
		x[i] =  (float) i /num_threads * 2.0f * PI;
		f[i] = 0.0f;
		v[i] = 0.0f;
	}

	for (i = 0; i < num_threads; i++) {
		rng_state[i] = (unsigned int)random();
	}

	cudaMalloc((void**)&drng_state, size2);
	cudaMalloc((void**)&dx, size);
	cudaMalloc((void**)&dv, size);
	cudaMalloc((void**)&df, size);

	cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dv, v, size, cudaMemcpyHostToDevice);
	cudaMemcpy(df, f, size, cudaMemcpyHostToDevice);
	cudaMemcpy(drng_state, rng_state, size2, cudaMemcpyHostToDevice);

	float tmp = 1.0 / sps;
	cudaMemcpyToSymbol(dt, &tmp, sizeof(float));

	gsl_histogram *h = gsl_histogram_alloc(BINS);
	gsl_histogram_set_ranges_uniform(h, 0.0f, 2.0f * PI);

	for (long step = 0; step < MAX_STEPS; step++) {
		float t = step / (float)sps;

		updateRHS<<<num_threads/64, 64, 64*sizeof(float)>>>(dx, dv);
		advanceSystem<<<num_threads/64, 64>>>(drng_state, dx, dv, df);

		if ((step+1) % OUTPUT_NTH == 0 && (output_histogram || output_avg)) {
			cudaMemcpy(x, dx, size, cudaMemcpyDeviceToHost);
			gsl_histogram_reset(h);

			double s = 0.0, c = 0.0;
			for (i = 0; i < num_threads; i++) {
				float tt = fmodf(x[i], (2.0f * PI));
				if (tt < 0.0f) {
					tt += 2.0f * PI;
				}
				s += sin(tt);
				c += cos(tt);
				gsl_histogram_increment(h, tt);
			}
			s /= num_threads;
			c /= num_threads;

			long hist_norm = 0;

			if (output_histogram) {
				for (i = 0; i < BINS; i++) {
					hist_norm += gsl_histogram_get(h, i);
				}

				for (i = 0; i < BINS; i++) {
					printf("%f %d %f\n", t, i,
							BINS*gsl_histogram_get(h, i)/(hist_norm*(hist_xmax-hist_xmin)));
				}
				printf("\n");
			}

			if (output_avg) {
				printf("%f %f %f %f\n", t, s, c, sqrt(s*s + c*c));
			}
		}
	}

	free(x);
	free(v);
	free(rng_state);

	cudaFree(drng_state);
	cudaFree(dx);
	cudaFree(dv);
}


