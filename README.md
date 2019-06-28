# Reducing Spreading Processes on Networks to Markov Population Models
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.org/gerritgr/Reducing-Spreading-Processes.svg?branch=master)](https://travis-ci.org/gerritgr/Reducing-Spreading-Processes)
## Overview
This is the official implementation to reproduce evaluation results in
[Reducing Spreading Processes on Networks to Markov Population Models](https://www.researchgate.net/publication/332607298_Reducing_Spreading_Processes_on_Networks_to_Markov_Population_Models).
This implementation does not contain the code to generate the reaction system/change vectors.


## Installation
If Python 3.6 and pip are installed, use
```sh
pip install -r requirements.txt
```
to install Python's dependencies.

## Usage
The results can be reproduces with: 
```sh
python evaluation.py
```
Moreover, an example on how to use **main.py** is given in the code. 


## Output
The **results** folder contains the summarized output. The **output** folder details information (graph file, full CTMC, numerical integration results, etc.).
