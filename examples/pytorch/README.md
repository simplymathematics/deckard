This folder contains a `dvc.yaml` file and a `conf/` directory that contain the reproducible steps. First, install this repository, which will include dvc as a dependency. Then, you can run:

```dvc repro```  
within one of the subdirectories of this directory. The `dvc.yaml` contains a series of "stages" that outline the dependencies, parameters, metrics, plots, and/or outputs of each stage. 
If an output from one stage is a dependency of another, then `dvc` will automatically know the order of operations, which is tremendously useful for development, allowing you to develop downstream changes and know that you can't accidentally delete weeks of calculations, though this requires setting up a dvc storage cache as per their docs.
You can view the dvc diagram with 
```dvc dag```
or by looking at the mermaid chart dag.md using a suitable rendering engine.
If you want to use a different pytorch-compatible model, modify the `torch-example.py` file and/or the configurations in the `conf/model/art` folder (for modifying training method or run-time parameters like optimizer)  or the `conf/model/initialize' to change the initialization parameters or file name for your custom model. 
