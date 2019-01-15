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
void convLayer_forward(int N, int M, int C, int H, int W, int K, float *X, float *W, float *Y) {
	int m, c, h, w, p, q;
	int H_out = H-K+1;
	int W_out = W-K+1;
	int n;	// for mini-batch

	//for(n=0; n<N; n++) 
	for(m=0; m<M; m++)  				// for each output feature maps
		for(h=0; h<H_out, h++)          // for each output element
			for(w=0; w<W_out; w++) {
				Y[m, h, w] = 0;
				//Y[n, m, h, w] = 0;
				for(c=0; c<C; c++) {	// sum over all input feature maps
					for(p=0; p<K; p++) {// K x K filter
						for(q=0; q<K; q++) {
							Y[m, h, w] += X[c, h+p, w+q] * W[m, c, p, q];
							//Y[n, m, h, w] += X[c, h+p, w+q] * W[m, c, p, q];
						}
					}
				}
			}
}

// sequential impl. of the forward propagation path of a subsampling layer. (POOLING)
// nonlinear functions: tanh, sigmoid, ReLU
void pollingLayer_forward(int M, int H, int W, int K, float *Y, float *S) {
	int m, h, w, p, q;
	for(m=0; m<M; m++) {				// for each output feature map
		for(h=0; h<H/K; h++) {			// for each output element
			for(w=0; w<W/K; y++) {
				S[m, h, w] = 0;
				for(p=0; p<K; p++) {	// loop over K x K input samples
					for(q=0; q<K; q++) {
						S[m, h, w] += Y[m, K*h+p, K*w+q]/(K*K);
					}
				}
				// add bias and apply non-linear activation
				S[m, h, w] = sigmoid(S[m, h, w] + b[m]);
			}
		}
	}
}

// dE/dX calculation of the backward path of a conv Layer
void convLayer_backward_xgrad(int M, int C, int H_in, int W_in, int K,
							  float *dE_dY, float *W, float *dE_dX) {
	int m, c, h, w, p, q;
	int H_out = H_in-K+1;
	int W_out = W_in-K+1;
	for(c=0; c<C; c++) 
		for(h=0; h<H_in; h++)
			for(w=0; w<W_in; w++)
				dE_dX[c, h, w]=0;
	
	for(m=0; m<M; m++) 
		for(h=0; h<H_out; h++) 
			for(w=0; w<W_out; w++) 
				for(c=0; c<C; c++) 
					for(p=0; p<K; p++) 
						for(q=0; q<K; q++) 
							dE_dX[c, h+p, w+q] += dE_dY[m, c, h, w] * W[m, c, p, q];
}

// dE/dW calculation of the backward path of a conv Layer
void convLayer_backward_wgrad(int M, int C, int H_in, int W_in, int K,
							  float *dE_dY, float *X, float *dE_dW) {
	int m, c, h, w, p, q;
	int H_out = H_in-K+1;
	int W_out = W_in-K+1;
	for(m=0; m<M; m++) 
		for(c=0; c<C; c++) 
			for(p=0; p<K; p++) 
				for(q=0; q<K; q++) 
					dE_dW[m, c, p, q] = 0;
	
	for(m=0; m<M; m++) 
		for(h=0; h<H_out; h++) 
			for(w=0; w<W_out; w++) 
				for(c=0; c<C; c++) 
					for(p=0; p<K; p++) 
						for(q=0; q<K; q++) 
							dE_dW[c, h+p, w+q] += X[c, h+p, w+q] * dE_dY[m, c, h, w];
}

/* After the dE_dW evaluated, weights are updated:
   w[t+1] = w[t]-lr*dE_dW
*/







