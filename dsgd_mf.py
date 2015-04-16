import sys
import numpy as np
import csv
from pyspark import SparkContext

class strata:
    def __init__(self, n):
        self.strata = range(n)
    def __iter__(self):
        return self
    def next(self):
        for i in xrange(len(self.strata)):
            if self.strata[i] + 1 == len(self.strata):
                self.strata[i] = 0
            else:
                self.strata[i] += 1
        return self.strata

def parseInput(line):
    """
    Parse each row into (mid, uid, rate) format.
    """
    tmp = line.split(",")
    return (int(tmp[0]), int(tmp[1]), float(tmp[2]))
    #return (int(tmp[1]), int(tmp[0]), float(tmp[2]))  ## only for autolab_train!

def parseInputNetflix(line):
    """
    Parse each row into (mid, uid, rate) format.
    """
    tmp = line[1].split()
    mid = int(tmp[0][:-1])
    ret = []
    for x in tmp[1:]:
        row = x.split(',')
        uid = int(row[0])
        rate = float(row[1])
        ret.append( (mid, uid, rate) )
    return ret

def parseInput2MapOfMap(line):
    """
    Parse each row into map(mid, map(uid, rate) ) format.
    """
    tmp = line[1].split()
    mid = int(tmp[0][:-1])
    ret = dict()
    for x in tmp[1:]:
        row = x.split(',')
        uid = int(row[0])
        rate = float(row[1])
        ret[uid] = rate
    return (mid, ret)

def divideBlocks(lst):
    """
    Divide a list into numWorker pieces.
    """
    ret = []
    for _ in xrange(numWorker):
        ret.append([])
    index = 0
    while index < len(lst):
        ret[index%numWorker].append(lst[index])
        index += 1
    return ret

def divideSets(lst):
    """
    Divide a list into numWorker pieces.
    """
    ret = []
    for _ in xrange(numWorker):
        ret.append(set())
    index = 0
    while index < len(lst):
        ret[index%numWorker].add(lst[index])
        index += 1
    return ret

def buildInvertedIndex(listOfSets):
    ret = dict()
    for i in xrange(len(listOfSets)):
        for elem in listOfSets[i]:
            ret[elem] = i
    return ret

def writeWandH():
    with open(outputWpath, 'wb') as wFile:
        writer = csv.writer(wFile)
        for mid in sorted(wk):
            writer.writerow(wk[mid])
    with open(outputHpath, 'wb') as hFile:
        writer = csv.writer(hFile)
        index = sorted(hk)
        for i in xrange(numFactor):
            writer.writerow([hk[x][i] for x in index])

def countPartitions(id, iterator):
    c = 0
    for _ in iterator:
        c += 1
    yield (id, c)

def showPartitions(id, iterator):
    c = []
    for x in iterator:
        c.append(x)
    yield (id, c)

def showPartitions2(id, iterator):
    c = np.zeros(numFactor)
    for x in iterator:
        c += wk[x]
    yield (id, c)

def main():
    rawData = sc.textFile(inputVpath, numWorker).map(parseInput)
    movieIndex = sorted(rawData.map(lambda x : x[0]).distinct().collect())
    userIndex  = sorted(rawData.map(lambda x : x[1]).distinct().collect())
    pMovieIndex = divideSets(movieIndex)
    pUserIndex  = divideSets(userIndex)
    invertedMovieIndex = buildInvertedIndex(pMovieIndex)
    rawData = rawData.map(lambda x : (invertedMovieIndex[x[0]], x)).partitionBy(numWorker).map(lambda x : x[1])
    rawData.cache()
    totalSelect = rawData.count()

    bUserIndex = sc.broadcast(pUserIndex)

    nI = rawData.map(lambda x : (x[0], 1)).reduceByKey(lambda a, b: a + b).collect()
    nJ = rawData.map(lambda x : (x[1], 1)).reduceByKey(lambda a, b: a + b).collect()
    bNI = sc.broadcast(dict(nI))
    bNJ = sc.broadcast(dict(nJ))
    bLambdaV = sc.broadcast(lambdaV)
    bMBetaV = sc.broadcast(-betaV)

    def showPartitions4(id, iterator):
        """
        for debug
        """
        ret = []
        for elem in iterator:
            if elem[1] in bUserIndex.value[curStrata[id]]:
                ret.append(elem)
        yield (id, [ bUserIndex.value[curStrata[id]], ret])

    #### Read data and broadcast done. ####

    def showStatus(id, iterator):
        """
        Show status of each worker in current strata.
        """
        ret = []
        wGrad = dict()
        hGrad = dict()
        for elem in iterator:
            if elem[1] in bUserIndex.value[curStrata[id]]:  #### only consider those data in this strata.
                mid = elem[0]
                uid = elem[1]
                rating = elem[2]
                predict = wk[mid].dot(hk[uid])
                error = rating - predict
                ret.append((mid, uid, rating, predict, error, numI.value[mid], numJ.value[uid]))
        yield (id, ret)

    def calRMSEByPartition(id, iterator):
        """
        Show status of each worker in current strata.
        """
        RMSE = 0.0
        for elem in iterator:
            if elem[1] in bUserIndex.value[curStrata[id]]:  #### only consider those data in this strata.
                mid = elem[0]
                uid = elem[1]
                rating = elem[2]
                predict = wk[mid].dot(hk[uid])
                RMSE += (rating - predict)**2
        yield (id, RMSE)

    def calRMSE(elem):
        """
        Show status of each worker in current strata.
        """
        RMSE = 0.0
        mid = elem[0]
        uid = elem[1]
        rating = elem[2]
        predict = wk[mid].dot(hk[uid])
        RMSE += (rating - predict)**2
        return RMSE

    def showRMSE(elem):
        """
        Show status of each worker in current strata.
        """
        RMSE = 0.0
        mid = elem[0]
        uid = elem[1]
        rating = elem[2]
        predict = wk[mid].dot(hk[uid])
        RMSE += (rating - predict)**2
        return (mid, uid, rating, predict, rating - predict)

    def updateBlock(id, iterator):
        """
        Calculate gradient in worker_id.
        """
        wGrad = dict()
        hGrad = dict()
        mbi = 0
        for elem in iterator:
            if elem[1] in bUserIndex.value[curStrata[id]]:  #### only consider those data in this strata.
                mid = elem[0]
                uid = elem[1]
                rating = elem[2]
                if mid not in wGrad:
                    wGrad[mid] = wk[mid]
                if uid not in hGrad:
                    hGrad[uid] = hk[uid]
                epsilon = 2*(100.0 + mbi + totalN) ** (bMBetaV.value)
                error = wGrad[mid].dot(hGrad[uid]) - rating
                tmpw = error*hGrad[uid] + (bLambdaV.value/bNI.value[mid])*wGrad[mid]
                tmph = error*wGrad[mid] + (bLambdaV.value/bNJ.value[uid])*hGrad[uid]
                wGrad[mid] -= epsilon*tmpw
                hGrad[uid] -= epsilon*tmph
                mbi += 1
        yield (id, [wGrad, hGrad, mbi])
        #yield (id, [wGrad, hGrad, mbi])

    #### Ininialize W and H matrix. ####
    global wk, hk
    wk = dict()
    hk = dict()

    for mid in movieIndex:
        wk[mid] = np.random.random(numFactor)
    for uid in userIndex:
        hk[uid] = np.random.random(numFactor)

    ### Ininialize strate array.
    myStrata = strata(numWorker)

    totalN = 0
    for t in xrange(numIteration):
        curStrata = myStrata.next()
    #    curRMSE = rawData.map(calRMSE).reduce(lambda a,b : a+b)
    #    print "total RMSE = ", curRMSE
        grads = rawData.mapPartitionsWithSplit(updateBlock).collectAsMap()
        for key, value in grads.iteritems():
            for k, v in value[0].iteritems():
                wk[k] = v
            for k, v in value[1].iteritems():
                hk[k] = v
            totalN += value[2]
        #print "t = {}, totalN = {}".format(t, totalN)
        curRMSE = rawData.map(calRMSE).reduce(lambda a,b : a+b) / totalSelect
        #print >> sys.stderr, "total RMSE = ", curRMSE
    writeWandH()

if __name__ == "__main__":
    np.random.seed(123)
    if len(sys.argv) != 9:
        print 'args error.'
        sys.exit(0)
    numFactor = int(sys.argv[1])
    numWorker = int(sys.argv[2])
    numIteration = int(sys.argv[3])
    betaV = float(sys.argv[4])
    lambdaV = float(sys.argv[5])
    inputVpath = sys.argv[6]
    outputWpath = sys.argv[7]
    outputHpath = sys.argv[8]

#sc = SparkContext('local', 'pyspark')
    setting = 'local[%d]' % numWorker
    sc = SparkContext(setting, 'pyspark')
    main()
