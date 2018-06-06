## Cyber Data Analytics - Lab 2 ##
Anomaly Detection in SCADA systems - BATADAL data set

Group 60 - Ernst Mulders and Alex Berndt

### Running the Code ###

- The code was written in Python 2.7.12
- Install the packages specified in the requirements.txt file
- You can use the requirements.txt to install the package requirements
- We recommend using a clean virtual environment to avoid possible package conflicts

##### Terminal Commands #####

- git clone https://github.com/alexberndt/CyberDataAnalytics/ Group60Lab2
- cd Group60Lab2/lab2/
- pip install -r requirements.txt
- python2 processTrainingDataset.py

The **main** file is ***processTrainingDataset.py***

### Analysis Methods ###

Anomaly detection is done by using different methods:
- **ARMA** time series predicition
- **N-gram** using SAX discretization and Markov chains
- Principle Components Analysis (**PCA**)

In ***processTrainingDataset.py***, change *analysisMethod* (line 99) to try the different methods. The ensemble methods can also be run by changing the *ensembleMethod* (line 100) variable.
