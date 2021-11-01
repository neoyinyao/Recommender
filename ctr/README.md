**Description:** 

an implementation of CTR prediction SOTA model working with tensorflow2.0

**Dataset:** 

criteo,  approximate 40 millions record.

**Implementation Detail:**

1.train_test_split: train ----> head 40,000,000 of total data(train.txt)  test ----> tail 5,840,617 of total data(train.txt), train.txt is already sorted by time.

**experiment result (see train log):** 

​						auc				bce_loss

[DLRM]: https://arxiv.org/pdf/1906.00091.pdf

​			0.8018			0.4534

[DeepFM]: https://arxiv.org/abs/1703.04247

 :		0.7849			0.4708

**TODO:**

implement DCN and DCNv2.

