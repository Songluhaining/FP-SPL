# VARCOP

### Ranking buggy SPL system

In order to rank buggy statements in SPL systems, you can simply configure the appropriate arguments in the file Main_BuggyStatements_Ranking.py and then execute it.

The meaning of arguments are as following:
1. **system_name**: For example Email, GPL, or ZipMe, etc
2. **buggy_systems_folder**: the path of the folder where you place the buggy versions of the systems, e.g. /Users/thu-trangnguyen/SPLSystems/Email/1Bug/4wise/
3. **sbfl_metrics**: The list of spectrum-based fault localization metrics that you would like to use for calculating suspiciousness scores of the statements
5. **normalization**: set it to be NORMALIZATION_ENABLE if you would like to normalize the suspicious scores of the statements in different variants, otherwise set it to be NORMALIZATION_DISABLE. The default value is NORMALIZATION_ENABLE.
6. **aggregation**: the aggregation formula that you want to use to aggregate scores of the statements in different variants, this argument can receive one of these values AGGREGATION_ARITHMETIC_MEAN, AGGREGATION_GEOMETRIC_MEAN, AGGREGATION_MIN, AGGREGATION_MAX, AGGREGATION_MODE, AGGREGATION_MEDIAN. The default value is AGGREGATION_ARITHMETIC_MEAN. 
7. **w**: the weight (from 0 to 1) used to combine product-based suspiciousness score and test case-based suspiciousness score. The default value is 0.5.

The ranking result of the buggy statements in each buggy version, which is ranked by VarCop, SBFL, S-SBFL, FB respectively, will be written into an excel file and store in the folder experiment_result, which is a folder of this project.

### Aggregating the average ranking result of VarCop, SBFL, S-SBFL, and FB
In order to aggregate the ranking results of the approaches, you can simply configure the appropriate arguments in the file Main_ExperimentalResultAnalysis.py and then execute it.

The meaning of arguments are as following:
1. **experimental_dirs**: The path of excel files containing ranking results that you want to aggregate
2. **num_of_examed_stms**: The number of statements that developers will investigate before giving up. This use to evaluate Hit@X, PBL. The default value is 10.


### Data.
1. The full version of data, you can download [here](https://tuanngokien.github.io/splc2021/)
2. For the sake of simplification, you can download [this](https://drive.google.com/drive/folders/19TrAf14FSdIkCVEQLjFGxlkkFcPSVR9Z?usp=sharing) light version, in which each case contains only spectrum files, files contain suspicious statements identified by VarCop and the slice-based method.

