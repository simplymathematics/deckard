# Default Workflow Flowchart

This page shows the default Deckard workflow as one high-level orchestration
diagram.

Shared semantic color system:

- Data -> green
- Model -> blue
- Attack -> red
- Detector -> purple
- Files -> light gray
- Scores -> orange
- Experiment -> dark gray

## Default Workflow

Canonical orchestration entrypoint: {meth}`deckard.experiment.ExperimentConfig.run`.

```mermaid
flowchart TD
    A[ExperimentConfig.run]
    B[Load and sample data]
    C[Optional pipeline transforms]
    D[Train or load model]
    E[Apply fit or predict defense]
    F[Baseline predictions]
    G[Optional attack branch]
    H[Optional detector branch]
    I[Score and aggregate metrics]
    J[Persist artifacts and outputs]

    A --> B --> C --> D --> E --> F --> I --> J
    E -. optional .-> G
    F -. optional .-> H
    G --> I
    H --> I

    classDef experiment fill:#e5e7eb,color:#111827,stroke:#4b5563,stroke-width:2px;
    classDef data fill:#dcfce7,color:#14532d,stroke:#22c55e,stroke-width:2px;
    classDef model fill:#dbeafe,color:#1e3a8a,stroke:#3b82f6,stroke-width:2px;
    classDef attack fill:#fee2e2,color:#991b1b,stroke:#ef4444,stroke-width:2px;
    classDef detector fill:#f3e8ff,color:#581c87,stroke:#a855f7,stroke-width:2px;
    classDef scores fill:#ffedd5,color:#9a3412,stroke:#fb923c,stroke-width:2px;
    classDef files fill:#f3f4f6,color:#374151,stroke:#9ca3af,stroke-width:2px;

    class A experiment;
    class B,C data;
    class D,E,F model;
    class G attack;
    class H detector;
    class I scores;
    class J files;
```

## Reading Guide

- The default path is `experiment -> data -> model -> scoring -> files`.
- Attack and detector stages are optional branches that feed back into the
  scoring stage.
- Persistence is the terminal step so downstream notebooks, plots, DVC stages,
  and reruns consume the same saved outputs.
