
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
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

1. Initiate an virtual environment (optional)
```
cd Online-Outlier-Detection
python -m virtualenv venv
 & ./venv/Scripts/Activate.ps1
```
2. Install the required libraries
```
pip install -r requirement.txt -U
```



<!-- USAGE EXAMPLES -->
## Usage
```
python OnlineAnomalyDetection.py -dataSource ./datasets/SPCTF-stream2D-withoutTimestamp.csv -targetColumn V45 -initialBatchTrainingSize 100 -batchSize 2 -maxSamples 100000 -timesteps_for_LSTM 5
```

<!-- CONTRIBUTING -->
## Contributing

### TODO





<!-- CONTACT -->
## Contact

### TODO



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements



