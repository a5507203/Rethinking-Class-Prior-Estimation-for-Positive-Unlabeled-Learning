
code for the ICLR2022 Paper "Rethinking Class-Prior Estimation for Positive-Unlabeled Learning"


+++++environment configuration++++++


#########important###############
The code is only tested on Linux based System (Ubuntu 20.04). 
The python version is 3.6.9. The pytorh version is 1.2.0 with GPU acceleration. 
#################################

The code for the AM and EN estimators were implemented in (Jain et al., 2016) and are acquired through personal communication. 
The code for ROC, KM1 and KM2 are taken from Clayton’s personal website (http://web.eecs.umich.edu/~cscott/). 

Beause some of the baselines are written in matlab  (i.e., AM, RAM, EN, REN, ROC, and RROC), to automate thousands of the experiments, 
we need to run matlab files in python and install some extra libraries. We apologize for the inconvenience.

Matlab is required to install. Note that, some extra matlab packages (e.g. deep learning toolbox) and python modules (e.g. matlab.engine) 
may need to be installed. Please install according to the hints (messages) showed on terminal when you run the code.

If you find that it is time-consuming to install matlab or extra matlab packages, you can simply comment out line 438 to line 462 
of "run_experiments.py". The trade-off is that the experiments will be only conducted on KM1, RKM1, KM2, RKM2, DPL, RDPL, RPG, and RRPG.



+++++run experiments++++++

The real word datasets are downloaded from the UCL machine learning database (https://archive.ics.uci.edu/ml/index.php). 
As mentioned in our paper, multi-class datasets are used as binary datasets by either grouping or ignoring classes.

#########run all experiments on a specific dataset with the fixed sample size################
Open a terminal at the project root directory and type the following command to run:

python3 run_experiments.py --dataset covtype_binary --sample_size 1600 --relabel_frac 0.1 > covtype_binary_relabelf0.1_1600.out

The results will be written in "covtype_binary_relabelf0.1_1600.out". It may take hours to finish this.

#########run all experiments################
We provide a simple shell script that allows to run all experiments with two line commands. 

Open a terminal at the project root directory and type the following commands:

sudo chmod 755 run_all.sh
./run_all.sh

If you have correctly configed the environment, all the experiments among all methods on the UCL databases will start to run.
It may take two-three days to finish all the experiments. You could exclude some of the experiments by modification of "./run_all.sh". 


References:
[1] Jain, S., White, M., Trosset, M. W., and Radivojac, P. Non-parametric semi-supervised learning of class proportions. arXiv preprint arXiv:1601.01944, 2016.
