**Description**: 

an Implementation of DIN(

[Deep]: https://arxiv.org/abs/1706.06978

) and DIEN(

[Deep]: https://arxiv.org/abs/1809.03672

) working with tensorflow2.0 ,inspired by alibaba official https://github.com/mouna99/dien.git which working with tensorflow1.14.

**Dataset:**

[AmazonBooks]: https://jmcauley.ucsd.edu/data/amazon/

**Implementation Detail:**

1. Dataset processing the same as https://github.com/mouna99/dien.git to reproduce paper
2. use tensorflow2.0 **AbstractCell**  implement **AUGRU** with mask, the same as **DynamicRNN**. because batch examples have **different** user behavior **length**;

**Experiment Result**:

​	auc : see train log

​		BASE: **0.7747**	

​		DIN: **0.7760**	

​		DIEN: **0.8209**

