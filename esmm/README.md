implementation of Alibaba Multi-Task cvr prediction Model: ESMM

paper: https://arxiv.org/pdf/1804.07931.pdf

data: https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

data processing copy https://github.com/xidongbo/AITM.git

experiment result:

​					Base 		ESMM 

CVR			0.6660 	0.6745

CTCVR		0.6419  0.6488

paper result:

​					Base 						ESMM 

CVR			66.00 ± 0.37			68.56 ± 0.37

CTCVR		62.07 ± 0.45			65.32 ± 0.49

detail: subsample non_click example in impression data to balance dataset, click:_num : non_click_num = 1:5