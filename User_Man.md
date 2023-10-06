# PEA-m6A User Manual (version 1.0)

- **PEA-m6A** is  an ensemble learning framework for predicting m6A modifications at regional-scale. PEA-m6A consists of four modules: **Sample Preparation, Feature Encoding, Model Development and Model Assessment**, each of which contains a comprehensive collection of functions with pre-specified parameters available.
- The PEA-m6A project is hosted on https://github.com/cma2015/PEA-m6A.
- The PEA-m6A Docker image can be obtained from https://hub.docker.com/r/malab/peam6a.
- The following part shows installation of PEA-m6A docker image and detailed documentation for each function in PEA-m6A.

## PEA-m6A installation

#### **Step 1**: Docker installation

**i) Docker installation and start ([Official installation tutorial](https://docs.docker.com/install))**

For **Windows (Only available for Windows 10 Prefessional and Enterprise version):**

- Download [Docker](https://download.docker.com/win/stable/Docker for Windows Installer.exe) for windows;
- Double click the EXE file to open it;
- Follow the wizard instruction and complete installation;
- Search docker, select **Docker for Windows** in the search results and click it.

For **Mac OS X (Test on macOS Sierra version 10.12.6 and macOS High Sierra version 10.13.3):**

- Download [Docker](https://download.docker.com/mac/stable/Docker.dmg) for Mac OS;

- Double click the DMG file to open it;
- Drag the docker into Applications and complete installation;
- Start docker from Launchpad by click it.

For **Ubuntu (Test on Ubuntu 18.04 LTS):**

- Go to [Docker](https://download.docker.com/linux/ubuntu/dists/), choose your Ubuntu version, browse to **pool/stable** and choose **amd64, armhf, ppc64el or s390x**. Download the **DEB** file for the Docker version you want to install;
- Install Docker, supposing that the DEB file is download into following path: *"/home/docker-ce~ubuntu_amd64.deb"*

```shell
  $ sudo dpkg -i /home/docker-ce<version-XXX>~ubuntu_amd64.deb      
  $ sudo apt-get install -f
```

**ii) Verify if Docker is installed correctly**

Once Docker installation is completed, we can run `hello-world` image to verify if Docker is installed correctly. Open terminal in Mac OS X and Linux operating system and open CMD for Windows operating system, then type the following command:

```
 $ docker run hello-world
```

**Note:** root permission is required for Linux operating system.

#### **Step 2**: PEA-m6A installation from Docker Hub

```shell
# pull latest PEA-m6A Docker image from docker hub
$ docker pull malab/peam6a
```

#### Step 3: Launch deepEA local server

```shell
$ docker run -it malab/peam6a bash
$ source activate
$ conda activate PEA-m6A
```

Then, PEA-m6A framework can be accessed.

## PEA-m6A Sample Preparation

This module provides two funcitons (see following table for details) to prepare epitranscriptome data.

| Functions                  | **Description**                                              | Input                                                        | Output                                                       |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| m<sup>6</sup>A-Seq process | read cleaning, read mapping, m6A peak calling, motif discovery | m<sup>6</sup>A-Seq data                                      | Clean reads in FASTQ format; Reads quality report in HTML format;Read alignments in SAM/BAM format; Peaks bed; Comprehensive overview of RNA modifications distribution in HTML or PDF format |
| Sample extraction          | Positive and negative samples extraction                     | Genome sequences; Peaks bed file; gene annotation(bed);exon annotation(bed) | Postive and negative sequences(fasta and bed)                |

#### m<sup>6</sup>A-Seq process

```python
python PEA-m6A.py data_process
```

#### Sample extraction

##### Input

- **Select species name**: by -s  
- **Select the path of annotation files**: by -ad
- **Select the file of peaks bed**: by -p
- **Select the output path**: by -o

##### Output

- Postive and negative sequences(fasta and bed)

#### How to use the function

```python
python PEA-m6A.py sample_extraction 
		-s Ath 
		-ad ../test_data 
		-p ../test_data/Ath_peaks.bed 
		-o ../output
```

## PEA-m6A Feature Encoding

This module provides three encoding strategies (see following table for details).

| Strategy             | Description                                                  | Input                     | Output                                                       |
| -------------------- | ------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------ |
| one-hot              | one-hot encoding was used to transfer nucleotides into numerical arrays: in which an *A* is encoded by [1,0,0,0], a *C* is encoded by [0,1,0,0], a *G* is encoded by [0,0,1,0], a *T* is encoded by [0,0,0,1]. | Sequences in FASTA format | Numerical arrays (can be used to train deep learning model)  |
| statistics-based     | statistic-based sequence features using *K*-mer (*K*-nucleotide frequencies) and *PseDNC* (pseudo-dinucleotide composition) algorithms | Sequences in FASTA format | Numerical array(42 * *N*, *N*  represents the number of sequences) |
| deep learning-dirven | deep learning-driven features can be extracted from nucleotide sequences using a weakly supervised learning framework WeakRM*, which was a modified version of WeakRM (Huang et al., 2021) | Sequences in FASTA format | Numerical arrays                                             |

#### Sample extraction

##### Input

- **Select species name**: by -s  
- **Select the path of fasta files**: by -i
- **Select the encoding strategy**: by -e
- **Select the output path**: by -o

##### Output

- Numerical arrays.

#### How to use the function

```python
## one-hot encoding strategy
python PEA-m6A.py features_encoding
		-s Ath
		-i ../test_data/Ath_test.fa
		-e onehot
		-o ../output
        
## statistics-based encoding strategy
python PEA-m6A.py features_encoding
		-s Ath
		-i ../test_data/Ath_test.fa
		-e statistics
		-o ../output
## deep learning-dirven encoding strategy
python PEA-m6A.py features_encoding
		-s Ath
		-i ../test_data/Ath_test.fa
		-e deeplearning
		-o ../output
```

## PEA-m6A Model Development

This module provides two funcitons (see following table for details) .

| Functions     | Description                                                 | Input                              | Ouput           |
| ------------- | ----------------------------------------------------------- | ---------------------------------- | --------------- |
| Model train   | Construct m6A modified prediction model                     | Train, valid and test datasets.    | Trained model   |
| Model predict | Select a model to make predictions about the input sequence | Numerical arrays and trained model | Predicted score |

#### Model train

##### Input

- **Select the path of train, valid and test datasets.**: by -i
- **Select the trained  strategy**: by -m
- **Select the output path**: by -o
- **Select the  model name**:by -cn

**Note:**The user can use the -matrix parameter to set the input feature matrix, PEA-m6A was specifically designed to accept feature matrix as the input for user customized feature encoding strategies.

##### Output

- Train, valid and test datasets.

#### How to use the function

```python
## train deep learnging model
python PEA-m6A.py train
		-i ../test_data
		-m WeakRM
		-o ../output
		-cn test_model
## train PEA-m6A model    
python PEA-m6A.py train
		-i ../test_data
		-m PEAm6A
		-matrix DL ST OT
		-o ../output
		-cn test_model
```

#### Model predict

##### Input

- **Select the path of test datasets.**: by -i
- **Select the trained  model name**: by -m
- **Select the output path**: by -o

##### Output

- Predicted score

#### How to use the function

```python
python PEA-m6A.py predict
		-i ../test_data
		-m model_name_trained
		-matrix DL ST OT
		-o ../output
```

## PEA-m6A Model Assessment

This module provides SHAP summary plot and SHAP Dependence Plot (see following table for details) to analysis model.**The SHAP summary plot** visualizes the influence of each feature on the model output, and **the SHAP Dependence Plot (SDP) **depicts the association between two different features.

#### SHAP summary plot

##### Input

- **Select the path of test datasets**: by -i
- **Select the trained  model name**: by -m
- **Select the plot name**: by -on
- **Select the output path**: by -o

##### Output

- The SHAP summary plot(pdf)

#### How to use the function

```python
python PEA-m6A.py model_analysis
		-i ../test_data
		-m model_name_saved
		-plot summary
		-on sumary_test
		-o ../output
```

#### SHAP Dependence plot

##### Input

- **Select the path of test datasets**: by -i
- **Select the trained  model name**: by -m
- **Select the first features name**: by -f1
- **Select the second features name**: by -f2
- **Select the plot name**: by -on
- **Select the output path**: by -o

##### Output

- The SHAP dependence plot(pdf)

#### How to use the function

```python
python PEA-m6A.py model_analysis
		-i ../test_data
		-m model_name_saved
		-plot dependence
		-on dependence_test
		-f1 features1_name
		-f2 faetures2_name
		-o ../output
```

