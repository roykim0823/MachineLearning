/* M: the number of output feature maps
   C: the number of input feature maps
   H: Height of each input map image
   W: Width of each input map image
   K: the height (and width) of each filter bank
   example)
   input X[C, H, W], output Y[M, H-K+1, W-K+1], W[C, M, K, K]
*/

// sequential impl. of the forward propagation path of a conv layer
// N is for minibatch size
/*
void convLayer_forward(int N, int M, int C, int H, int W, int K, float *X, float *W, float *Y) {
	int m, c, h, w, p, q;
	int H_out = H-K+1;
	int W_out = W-K+1;
	int n;	// for mini-batch

	parallel_for(n=0; n<N; n++) 
	parallel_for(m=0; m<M; m++)  				// for each output feature maps
		parallel_for(h=0; h<H_out, h++)          // for each output element
			parallel_for(w=0; w<W_out; w++) {
				Y[m, h, w] = 0;
				//Y[n, m, h, w] = 0;
				for(c=0; c<C; c++) {	// sum over all input feature maps
					for(p=0; p<K; p++) {// K x K filter
						for(q=0; q<K; q++) {
							Y[m, h, w] += X[c, h+p, w+q] * W[m, c, p, q];
							//Y[n, m, h, w] += X[n, c, h+p, w+q] * W[m, c, p, q];
						}
					}
				}
			}
}
*/

/* CUDA NAIVE version:
   - 2D thread blocks, with each block computing a tile of TILE_WIDTH x TILE_WIDTH
   - Excessive global memory accesses!
*/
__global__ void convLayerForward_Kernel1(int C, int W_grid, int K, float *X, float *W, float *Y) {
	int n, m, h, w, c, p, q;
	n = blockIdx.x;
	m = blockIdx.y;
	h = blockIdx.y / W_grid + threadIdx.y;
	w = blockIdx.z % W_grid + threadIdx.x;
	float acc=0;
	for(c=0; c<C; c++) 
		for(p=0; p<K; p++) 
			for(q=0; q<K; q++) 
				acc += X[n, c, h+p, w+q] * W[m, c, p, q]; 
	
	Y[n, m, h, w] = acc;
}

/* CUDA Improved using shared memory:
   1. Load the filter W[m, c] into the shared memory
   2. All threads collaborate to copy the portion of the input X[n, c, ...] that is requried to
      compute the output tile into the shared memory array X_shared
   3. Compute the partial sum of output Y_shared[n, m, ...]
   4. Move to the next input channel c
*/
__global__ void convLayerForward_Kernel2(int C, int W_grid, int K, float *X, float *W, float *Y) {
	int n, m, h0, w0, h_base, w_base, h, w;
	int X_tile_width = TILE_WIDTH + K-1;
	extern __shared__ float shmem[];
	float *X_shared = &shmem[0];
	float *Y_shared = &shmem[X_tile_width * X_tile_width];
	n=blockIdx.x;		m = blockIdx.y; 
	h0 = threadIdx.x;	w0 = threadIdx.y;	// h0 and w0 used as shorthand for threadIdx.x and .y
	h_base = (blockIdx.z / W_grid) * TILE_SIZE;		// vertical base out data index for the block
	w_base = (blockIdx.z % W_grid) * TILE_SIZE;		// horizontal base out data index for the block
	h = h_base + h0;
	w = w_base + w0;

	float acc=0;
	int c, i, j, p, q;
	for(c = 0; c<C c++) { 	// sum over all input channels
		if( (h0 < K) && (w0 < K) ) 
			W_shared[h0, w0] = W[m, c, h0, w0];	// load weights for W[m, c, ...]
		__syncthreads();						

		for(i=h; i<h_base+X_tile_width; i+=TILE_WIDTH) {
			for(j=w; j<w_base+X_tile_width; j+=TILE_WIDTH) 
				X_shared[i-h_base, j-w_base] = X[n, c, h, w];	// load input
		}
		__syncthreads();

        for(p=0; p<K; p++) {
			for(q=0; q<K; q++) 
				acc += X_shared[h+p, w+q] * W_shared[p, q];
		}
		__syncthreads();
	}
	Y[n, m, h, w] = acc;
}

/* Final CUDA version using GEMM (GEneral Matrix to Matrix Multiplication):
   - Central Idea is unfolding and replicating the inputs to the convolutional kernel such that
     all elements needed to compute one output element will be stored as one sequential block
   - This will reduce the forward operation of the conv layer to one large Matrix-Matrix Mult.
*/

void convLayer_forward(int N, int M, int C, int H, int W, int K, float *X, float *W_unroll, float *Y)
{
	int W_out = W-K+1;
	int H_out = H-K+1;
	int W_unroll = C*K*K;
	int H_unroll = H_out*W_out;
	float *X_unrolled = malloc(W_unroll * H_unroll * sizeof(float));
	for(int n=0; n<N; n++) {
		unroll(C, H, W, K, n, X, X_unroll);
		gemm(H_unroll, M, W_unroll, X_unrolled, W, Y[n]);
	}
}

void unroll(int C, int H, int W, int K, float *X, float *X_unroll) {
	int c, h, w, p, q, w_base, w_unroll, h_unroll;
	int H_out = H-K+1;
	int W_out = W-K+1;
	
	for(c=0; c<C; c++) {
		w_base = c*(K*K);
		for(p=0; p<K; p++) 
			for(q=0; q<K; q++) {
				for(h=0; h<H_out; h++) 
					for(w=0; w<W_out; w++) {
						w_unroll = w_base + p*K+q;
						h_unroll = h*W_out+w;
						X_unroll[h_unroll, w_unroll] = X[c, h+p, w+q];
					}
			}
	}
}

void unroll_gpu(int C, int H, int W, int K, float *X, float *X_unroll) {
	int H_out = H-K+1;
	int W_out = W-K+1;
	int num_threads = C*H_out, W_out;
	int num_blocks = ceil( (C*H_out*W_out)/ CUDA_MAX_NUM_THREADS);
	unroll_Kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>();
}

__global__ void unroll_Kernel(int C, int H, int W, int K, float *X, float *X_unroll) {
	int c, s, h_out, w_out, h_unroll, w_base, p, q;
	int t= blockIdx.x*CUDA_MAX_NUM_THREADS+threadIdx.x;
	int H_out = H-K+1;
	int W_out = W-K+1;
	int W_unroll = H_out * W_out;

	if(t < C * W_unroll) {
		c= t/W_unroll;
		s= t%W_unroll;
		h_out = s/W_out;
		w_out = s%W_out;
		h_unroll = h_out * w_out;
		w_base = c*K*K;
		for(p=0; p<K; p++) 
			for(q=0; q<K; q++) {
				w_unroll = w_base + P*K +q;
				X_unroll[h_unroll, w_unroll] = X[c, h_out+p, w_out+q];
			}
	}
}

				

