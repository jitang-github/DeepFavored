# DeepFavored
A deep learning based method to identify favored (adaptive) mutations in population genomes.

This method is described in the paper: [Ji Tang, Maosheng Huang, Sha He, Junxiang Zeng, Hao Zhu (2022). Uncovering the extensive trade-off between adaptive evolution and disease susceptibility. Cell Reports, Volume 40, Issue 11, 111351](https://doi.org/10.1016/j.celrep.2022.111351). Please cite it if you use our method.

By applying this method to human genomes and performing statistical analysis of the output and GWAS datasets, we got interesting findings. Check the paper above for these findings and the study details. In addition, *The Scientist* magzine introduced these findings, along with discussions, to general audiences, check out it at [Historic Adaptations May Now Make Us Susceptible to Disease](https://www.the-scientist.com/historic-adaptations-may-now-make-us-susceptible-to-disease-70506).

## Contents
This directory contains the following:

    1. DeepFavored.py - interface for training and using DeepFavored
    2. train.py
    3. identify.py
    4. network.py 
    5. utils.py
    6. hyperparams.json
    7. requirements.txt - list of the dependencies
    8. example/ - example data for training and using DeepFavored
            training_data/ - example training data
                sweep_regions/
                    500000_rep0
                neutral_regions/
                    rep0
            test_data/ - example test data for evaluating the performance of trained DeepFavored
                mht_plot_data/
                    159174683_rs2814778_DARC_YRI
                rankCDF_powerFDR_data/
                    neutral_rep0
                    sweep_500000_rep0
            regions_to_be_identified/ - example data to be identified using trained DeepFavored
                chr1-158674719-159674668_DARC_YRI


## Installation
```
$conda create -n deepfavored python==3.6.12
$conda activate deepfavored
$pip install -r requiredments.txt
```


## Training DeepFavored
### Command Line
    $python DeepFavored.py train
#### Options
    --hyparams \<string\>: Required, path to the file documenting hyper parameters for training
    
    --trainData \<string\>: Required, path to the training data directory
    
    --modelDir \<string\>: Required, path to the directory saving trained model and related files
    
    --testData \<string\>: Optional, path to the test data directory

### Training data
#### Directory hierarchy
The following directories must be in a single directory, the path to which will be passed to *DeepFavored.py train* using the flag *--trainData*. Subdirectories are allowed.
>     sweep_regions/
>     neutral_regions/

#### Directory introduction
1. sweep_regions/: loading sweep regions' files. Each file should include one favored mutation site.
2. neutral_regions/: loading neutral regions' files.
   
#### File format
1. All files under directory *sweep_regions* are tab-delimited. In each file, as shown by the following table, the first column should be position of a mutation site(the column name should be *"POS"*), the second column should be a mutation site's LD with favored mutation site(the column name should be *"LD"*), the subsequent columns should include each of the component 
statistics specified in *hyparams.json*(column names should exactly be the same as the component statistics), if necessary additional columns for identification can follow. The order of the columns for component statistics does not matter.
2. All files under directory *neutral_regions* have the same format as the files under directory *sweep_regions*, except no need to include the LD column

|POS  | LD  | Fst  |XPEHH |  iHS |DDAF  | iSAFE|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1   | 0.6  | 3.4  |  2.2 | -3.4 |  1.3 | 0.1  |
| ...  | ...  | ...  |  ... | ...  |  ... | ...  |

#### File name
1. Under directory *sweep_regions*, all files' names need to be *"_"* delimited, and the first term is the position of the favored mutation inside the file, if necessary additional terms for identification can follow.
2. No requirement for the name of the files under directory *neutral_sweep*

### Test data(optional)
Test data are sweep and neutral regions, as what we encounter when identifying favored mutations in real population genomes.
When test data path is provided by *--testData*, the performance of the trained model will be automatically evaluated with the test data once training process is finished, plotted as manhattan plot, rankCDF curve or powerFDR curve.
The optimal values of hyper parameters in file *hyparams.json* were found and evaluated on European, African and EastAsian populations, it is recommended to provide test data when applying DeepFavored to other human populations or other species' populations.

#### Directory hierarchy
The following directories must be in a single directory, the path to which will be passed to *DeepFavored.py train* using the flag *--testData*. Subdirectories are allowed.
>     mht_plot_data/
>     rankCDF_powerFDR_data/

#### Directory introduction
1. mht_plot_data/: In your situation, there may be several well-characterized sweep regions whose favored mutation's position is known. If you offer them here, all the SNPs' DFscores will be drawn as manhattan plot, in which the DFscore of favored SNP(specified in file name) will be marked as red triangle.
2. rankCDF_powerFDR_data/: The data here are hundreds or above of sweep or neutral regions, in general produced by simulation. To evaluate the performance, rank-cumulative distribution function(CDF) curve will be plotted with the DFscores of the sweep regions, and power-false discovery rate(FDR) curve will be plotted with the DFscores of the sweep and neutral regions.

#### File format
All the files under directories *mht_plot_data* and *powerFDR_curve_data* have the same format as the files under the directory *neutral_sweep* mentioned above.

#### File name
1. Under directory *mht_plot_data*, all files' names need to be *"_"* delimited, and the first term is the position of the favored mutation inside the file, if necessary additional terms for identification can follow.
2. Under directory *powerFDR_curve_data*, all files' names need to be *"_"* delimited. For a sweep region file, the first term should be the string *"sweep"*, the second term should be position of the favored mutation site inside the file, and if necessary additional terms for identification can follow. For a neutral region file, the first term should be string *"neutral"*, and if necessary additional terms for identification can follow.


### Output
Under the directory specified by *--modelDir*, *DeepFavored.py train* output the following:

    1. model/: directory saving trained model
    2. performance/: directory saving the performance evaluation of the trained model performed on the test data specified by *--testData* option. When the test data is not provided, this directory does not appear
        (1) mht_plot/ : directory saving manhattan plots produced with the data under the *mht_plot_data* directory mentioned above
        (2) rankCDFcurve.png : rank-CDF curve produced with the sweep regions under the *rankCDF_powerFDR_data* directory mentioned above
        (3) powerFDRcurve.png : power-FDR curve produced with the data under the *rankCDF_powerFDR_data* directory mentioned above
        (4) confusionMtxNumbers.tsv : confusion matrix numbers produced with the sweep and neutral regions under the *rankCDF_powerFDR_data* directory mentioned above. At here, you can tangibly see the number of the favored mutations identified correctly or wrongly
    3. hyperparams.json: file saving hyper-parameters of the trained model
    4. training.log: file logging the information produced during training process


### Example
```
$ python DeepFavored.py train --hyparams ./hyperparams.json
                    --trainData ./example/training_data
                    --testData ./example/test_data
                    --modelDir ./example/example.df.model
```


## Using DeepFavored
### Command Line
    $python DeepFavored.py identify
#### Options
    --model \<string\>: Required, path to the directory saving trained model, i.e the directory specified by *--modelDir* above
    
    --input \<string\>: Required, path to a file or a directory. the file or the files under the directory have the same format as the files under the directory "neutral_sweep" mentioned above, No requirement for file name.
    
    --outDir \<string\>: Optional, path to output directory. If not specified, it will be the same direcory as the input by default


### Output
Each output file(suffixed by *.df.out*) is a TAB separated file in the following format:

| POS  | DF_H | DF_O |  DF  |
|:----:|:----:|:----:|:----:|
| 102  | 0.9  |  0.8 | 0.72 |
| ...  | ...  |  ... | ...  |

With the following headers:
- POS: Position (bp) sorted in ascending order
- DF_H: Score output by the 'H' classifier
- DF_O: Score output by the 'O' classifier
- DF: DeepFavored score(DFscore for short)


### Example
```
$ python DeepFavored.py identify --modelDir ./example/example.df.model
                      --input ./example/regions_to_be_identified
                      --outDir ./example/regions_identified_by_df
```
