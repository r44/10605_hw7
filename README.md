## Description

* Hw7 of 10-605, 
* Due Date: 4/16/2015
* Author: Jui-Pin Wang

## How to Run
* Run autolab_train.csv
```
$ $(SPARK) dsgd_mf.py $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TRAINV) $(OUTPUTW) $(OUTPUTH)  
```
For example,
```
$ spark-submit dsgd_mf.py 20 5 100 0.9 0.1 autolab_train.csv w.csv h.csv  
```

