sourcefiles_core = \
code/src/core/Attribute.cpp \
code/src/core/DenseInstance.cpp \
code/src/core/DiscreteEstimator.cpp \
code/src/core/DoubleVector.cpp \
code/src/core/GaussianEstimator.cpp \
code/src/core/Instance.cpp \
code/src/core/InstanceInformation.cpp \
code/src/core/SparseEstimator.cpp \
code/src/core/SparseInstance.cpp \
code/src/core/SplitCriterion.cpp \
code/src/utils/Utils.cpp \
code/src/utils/Log.cpp \
code/src/utils/CommandLineParser.cpp \
code/src/utils/Configurable.cpp \
code/src/utils/DynamicCreateClass.cpp \
code/src/utils/jsoncpp.cpp \
code/src/utils/LearnerModel.cpp \
code/src/utils/RTTI.cpp \
code/src/streams/Reader.cpp \
code/src/streams/ArffReader.cpp \
code/src/streams/C45Reader.cpp \
code/src/streams/CSVReader.cpp \
code/src/streams/LibSVMReader.cpp \
code/src/evaluation/BasicClassificationEvaluator.cpp \
code/src/evaluation/Evaluator.cpp \
code/src/evaluation/Measures.cpp \
code/src/evaluation/NullEvaluator.cpp \
code/src/learners/Learner.cpp \
code/src/learners/Classifiers/Functions/Logisticregression.cpp \
code/src/learners/Classifiers/Functions/Majorityclass.cpp \
code/src/learners/Classifiers/Functions/Perceptron.cpp \
code/src/learners/Classifiers/Bayes/NaiveBayesStatistics.cpp \
code/src/learners/Classifiers/Bayes/Naivebayes.cpp \
code/src/learners/Classifiers/Bayes/SimpleNaiveBayesStatistics.cpp \
code/src/learners/Classifiers/Meta/Bagging.cpp \
code/src/learners/Classifiers/Bayes/observer/NominalAttributeClassObserver.cpp \
code/src/learners/Classifiers/Bayes/observer/NumericAttributeClassObserver.cpp \
code/src/learners/Classifiers/Trees/ADWIN.cpp \
code/src/learners/Classifiers/Trees/AttributeSplitSuggestion.cpp \
code/src/learners/Classifiers/Trees/HATNode.cpp \
code/src/learners/Classifiers/Trees/HoeffdingAdaptiveTree.cpp \
code/src/learners/Classifiers/Trees/HoeffdingTree.cpp \
code/src/learners/Classifiers/Trees/HTAttributeClassObserver.cpp \
code/src/learners/Classifiers/Trees/HTNode.cpp \
code/src/learners/Classifiers/Trees/InstanceConditionalTest.cpp \
code/src/tasks/EvaluatePrequential.cpp \
code/src/tasks/EvaluateHoldOut.cpp \
code/src/tasks/Task.cpp \
code/src/main.cpp


sourcefiles = $(sourcefiles_core) 

targetfile = streamdm

includepath = -Icode/src/core -Icode/src/learners  -Icode/src \
    -Icode/src/utils -Icode/src/learners/classifiers/Bayes \
    -Icode/src/tasks -Icode/src/learners/classifiers/Trees \
    -Icode/src/learners/Classifiers/Bayes/observer/ \
    -Icode/src/learners/Classifiers/Meta/ \
    -Icode/src/learners/Classifiers/Functions/ \
    -Icode/src/evaluation/ \
    -Icode/src/streams/ 
    

glog = -lglog -lpthread

log4cxx = -llog4cxx

log4cpp = -llog4cpp -lpthread

debug = -g

warning = -Wall -Wno-sign-compare -Wno-nonnull -Wno-unused-variable

#tesla:cc61
acc_flags = -acc -Minfo=accel -Mcuda=cuda11.4,rdc -ta=tesla,cc61 -DUSE_ATOMIC

flags = -std=c++11 -O3 -lpthread -silent $(acc_flags) #$(debug)  

all:
	/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/compilers/bin/pgc++ $(sourcefiles) $(includepath) $(log4cpp) $(flags) -o $(targetfile)


