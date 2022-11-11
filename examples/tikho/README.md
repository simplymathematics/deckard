## Params
All of the run-time params are specified in the the conf/ folder. The default configurations are specified in config.yaml.

## Pipeline
The pipeline is described in dvc.yaml, including the metrics, plots, outputs, dependencies, commands, and parameters as specified in that file.

## Dependency graph

To see how each `stage` of said pipeline relates to each other run `dvc dag`.
To see the all of the file system dependencies (the graph below), run: `dvc dag -o --md`.
Files:
```mermaid
flowchart TD
	node1["data/data.npz"]
	node2["data/ground_truth.csv"]
	node3["experiments"]
	node4["model/model.pickle"]
	node5["model/predictions.csv"]
	node6["model/scores.csv"]
	node7["params.yaml"]
	node8["report/balance.png"]
	node9["report/classification.png"]
	node10["report/confusion.png"]
	node11["report/correlation.png"]
	node12["report/radviz.png"]
	node13["report/report.html"]
	node1-->node3
	node1-->node4
	node1-->node5
	node1-->node6
	node1-->node9
	node1-->node10
	node1-->node13
	node2-->node3
	node4-->node3
	node5-->node3
	node7-->node1
	node7-->node2
	node7-->node4
	node7-->node5
	node7-->node6
	node7-->node8
	node7-->node9
	node7-->node10
	node7-->node11
	node7-->node12
	node7-->node13
	node8-->node3
	node9-->node3
	node10-->node3
	node11-->node3
	node12-->node3
	node13-->node3
```
