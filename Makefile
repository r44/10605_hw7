DATAPATH=/afs/cs.cmu.edu/project/bigML/
INPUTV=$(DATAPATH)netflix-ratings/training_set
TESTV=../testV
TESTV1=../testcase1.csv
TESTV2=../testcase2.csv
TESTV3=../testcase3.csv
TRAINV=../autolab_train.csv
OUTPUTW=./w.csv
OUTPUTH=./h.csv
LOG1=./spark_dsgd.log
LOG2=./eval_acc.log

#NUM_FACTOR=20
NUM_FACTOR=20
NUM_WORKER=5
NUM_ITER=100
BETA=0.9
LAMBDA=0.1

SPARKPATH=/usr/local/Cellar/spark-1.3.0-bin-hadoop2.4/bin/
SPARK=$(SPARKPATH)spark-submit 
MAIN=dsgd_mf.py

val:
		python eval2.pyc $(LOG2) $(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TRAINV) $(OUTPUTW) $(OUTPUTH) > $(LOG1)
eval:
		python eval_acc.py $(LOG2) $(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TRAINV) $(OUTPUTW) $(OUTPUTH)  

test:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TESTV) $(OUTPUTW) $(OUTPUTH)  

train:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TRAINV) $(OUTPUTW) $(OUTPUTH) 2> /dev/null 

t1:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TESTV1) $(OUTPUTW) $(OUTPUTH)  

t2:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TESTV2) $(OUTPUTW) $(OUTPUTH)  

t3:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(TESTV3) $(OUTPUTW) $(OUTPUTH)  


clean:
	rm -f *.class
tar:
	tar -cvf hw7.tar *.py *.csv *.log juipinw-report.pdf
	#tar -cvf hw7.tar *.py *.csv eval_acc.log juipinw-report.pdf

#	pyspark $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(INPUTV) $(OUTPUTW) $(OUTPUTH)  
run:
	$(SPARK) $(MAIN) $(NUM_FACTOR) $(NUM_WORKER) $(NUM_ITER) $(BETA) $(LAMBDA) $(INPUTV) $(OUTPUTW) $(OUTPUTH)  
