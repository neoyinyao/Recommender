#  implementation of PinSage working with dgl Deep Graph Library and Tensorflow

Inspired by https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage

**Implementation Detail:**

code realization corresponding with raw paper: 

1. On-the-fly convolution, DGL PinSage Sampler form local graph neighborhoods, only local graph node and feature load into GPU. 
2. Map-Reduce inference, build PySpark demo, not so efficient because of python object serializaiton and deserializaiton, but idea is same. 

3. Efficient random walks with DGL PinSage Sampler. 

4. Importance Pooling,realize Convolve algorithm.

5. train with mini-batch manner.

6. train with subgraph, inference in total graph.

   

**TODO:**

1. DGL with Tensorflow static computation graph raise error,only run in eager mode. 

2. train is synchronous,no Produce-Consumer architecture. 

3. train is local, no distributed training. 

4. no curriculum training,no hard negative example selection.