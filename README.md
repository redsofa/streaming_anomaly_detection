
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!--
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    -->
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Online Anomaly Detection with LOF, HS-Tree, IFOREST, AE-LSTM

### Built With

Major frameworks that we are using.
* [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/index.html#)
* [Tensorflow](https://www.tensorflow.org/)
* [tdigest](https://github.com/CamDavidsonPilon/tdigest)



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites
* Python 3.7+

### Installation

1. Windows
```
python -m virtualenv venv
 & ./venv/Scripts/Activate.ps1
pip install -r requirement.txt -U
```


2. Mac OS X
```
pip install virtualenv
python -m virtualenv venv
source ./venv/bin/activate
pip install -r requirement.txt -U
```

3. Linux
```
pip install virtualenv
python -m virtualenv venv
source ./venv/bin/activate
pip install -r requirement.txt -U
```

## Usage Example on Shuttle data - (OS X and Linux)
```
cd ./source
mkdir data_results
python OnlineAnomalyDetection.py -dataSource ../datasets/shuttle_small.csv -targetColumn 9 -initialBatchTrainingSize 100 -batchSize 5 -maxSamples 10000 -timesteps_for_LSTM 5
```

## Citation
If you use or reference this work in a scientific publication,
we would appreciate that you use the following citations:

```
@INPROCEEDINGS{belacel2022lstmae,
  title={An LSTM Encoder-Decoder Approach for Unsupervised Online Anomaly Detection in Machine Learning Packages for Streaming Data},
  author={Belacel, Nabil and Richard, Rene and Xu, Zhicheng},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={},
  year={2022},
  organization={IEEE}
}
```
