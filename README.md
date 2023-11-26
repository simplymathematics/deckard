

# Deckard: A Tool for Evaluating AI

## 1 - Dependencies

To install this, ensure that you have your favorite library installed. To install deckard along with `tensorflow`, for example, use
```
python -m pip install .[tensorflow]
```
Add the `-e` flag if you want to edit files:
```
python -m pip install -e .
```
Or try the rpi script:
```
bash rpi.sh
```
Now, check that deckard works
```$ python```
```>>> import deckard```
Then CTRL+D or `quit()` to quit.
##  2 - Navigate to your favorite subfolder in `examples`. One is provided for each framework.
Running `dvc repro` in that folder will reproduce the experiment outlined in the `dvc.yaml`. Running 
```python -m deckard```
 will parse the configuration folder, create a `params.yaml`, and then run `dvc repro`.
### _like tears in the rain_, this tool is meant for bladerunners. NOT INTENDED FOR USE BY REPLICANTS

## Files

.  
├── Dockerfile: Constructs a generic Docker image for running experiments  
├── LICENSE  
├── README.md: this file  
├── deckard: Source code  
├── examples: Directory containing all the examples  
├── rpi.sh: For installation on Raspbian.  
├── setup.py : for installation with pip  
├── setup.sh : for installation using bash  
└── test : test suite  
 

###

To build the package (optional and is a very rough draft):

```


After adding it to your path, you can then run it as a module:

```
python -m deckard --config_name mnist.yaml --config_folder examples/power
```
