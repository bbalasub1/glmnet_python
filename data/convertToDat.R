# convert RData to csv
rm(list = ls())
baseDir = '/home/bbalasub/Desktop/Summer2016/glmnet/github/glmnet_python/data/'
fileName = 'CoxExample'

fullFileName = paste(baseDir,fileName, '.RData',sep = '');
load(fullFileName)
ls()
dim(x)
dim(y)
outfileNameX = paste(fileName, 'X.dat', sep='');
outfileNameY = paste(fileName, 'Y.dat', sep='');
write.table(x, file = outfileNameX,row.names=FALSE, na="",col.names=FALSE, sep=",")
write.table(y, file = outfileNameY,row.names=FALSE, na="",col.names=FALSE, sep=",")
