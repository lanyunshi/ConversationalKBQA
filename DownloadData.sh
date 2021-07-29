mkdir CONVEX
mkdir CONVEX/data
wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_train.zip -P CONVEX/data
unzip CONVEX/data/ConvQuestions_train.zip -d CONVEX/data
wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_dev.zip -P CONVEX/data
unzip CONVEX/data/ConvQuestions_dev.zip -d CONVEX/data
wget http://qa.mpi-inf.mpg.de/convex/ConvQuestions_test.zip -P CONVEX/data
unzip CONVEX/data/ConvQuestions_test.zip -d CONVEX/data
