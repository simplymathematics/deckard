# %%
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from joblib import delayed, Parallel
import matplotlib.pyplot as plt

def find_mean_std(group, val_cols, data):
    for col in val_cols:
        subset = data.loc[group.index]
        mean = subset[col].mean()
        std = subset[col].std()
        group[col + "_mean"] = mean
        group[col + "_std"] = std
    return group

def find_column_subset(keep_these, data):
    column_mask = []
    for col in keep_these:
        if col not in data.columns:
            for c in data.columns:
                if col in c:
                    column_mask.append(c)
        else:
            column_mask.append(col)
    data = data[column_mask]
    return data

def group_data(do_not_group, group_these, data):
    group_by = []
    for col in group_these:
        if col not in data.columns:
            for c in data.columns:
                if col in c:
                    group_by.append(c)
        else:
            group_by.append(col)
    group_these = group_by
    for col in do_not_group:
        if col in group_these:
            group_these.remove(col)
    print(f"Grouping with {group_these}")
    grouped = data.groupby(group_these)
    return grouped

def process_groups(val_cols, do_not_group, group_these, data):
    grouped = group_data(do_not_group, group_these, data)
    results = Parallel(n_jobs=-1)(
            delayed(find_mean_std)(group, val_cols, data) 
            for _, group in tqdm(grouped)
        )
    new_data = pd.concat([*results])
    return new_data


if __name__ == "__main__":
    # %%
    #  Specify some input files
    input_file = "output/combined/plots/search_merged.csv"
    hamming_file = "output/combined/plots/hamming_merged.csv"
    output_file = "output/combined/plots/search_averaged.csv"
    # Measured variables
    val_cols = [
        "accuracy",
        "predict_time",
        "train_time",
        "train_time_per_sample",
        "predict_time_per_sample",
    ]
    # Excluding these because we want to average over each fold,
    # and each fold has unique train/test matrices
    do_not_group = [
        "model.init.distance_matrix_train",
        "model.init.distance_matrix_test",
    ]
    group_these = ["Dataset", "Model", "Metric", "algorithm", "model.init"]
    keep_these = group_these + val_cols

    data = pd.read_csv(input_file, index_col=0)
    hamming_data = pd.read_csv(hamming_file, index_col=0)
    data = pd.concat([data, hamming_data], axis = 0 )

    # Some data cleaning
    # fillna with 0 because nans confuse the groupby
    data = data.fillna(0)
    # Bigger numbers. Yay!
    data["accuracy"] = data["accuracy"] * 100

    # Keeps anything in keep_these
    data = find_column_subset(keep_these, data)
    if not Path(output_file).exists():
        # Calculates the mean, std for each value in each group
        new_data = process_groups( val_cols, do_not_group, group_these, data)
        new_data['Kernel or Distance'] = new_data.apply(lambda row: "Kernel" if not row['model.init.transform'] == "D" else "Distance", axis=1)
        new_data['Hamming or RBF'] = new_data.apply(lambda row: "RBF" if any(x in str(row['model.init.transform']) for x in ["exp_neg_", "rbf_"]) else ("Hamming" if "hamming" in str(row['model.init.transform']) else None), axis=1)
        new_data['Compressor or String Metric'] = new_data.apply(lambda row: "Compressor" if row['Metric'] in ["GZIP","Brotli", "BZ2"] else "String Metric", axis=1)
        data = new_data
        new_data.to_csv(output_file)
        
    else:
        data = pd.read_csv(output_file, index_col=0)

    ########################################################################################
    # %%
    sns.set_theme(context="paper", style="whitegrid", font="Times New Roman", font_scale=2)

    plt.gcf().clear()
    acc_graph1 = sns.barplot(
        data=data,
        hue='Kernel or Distance',
        x="Metric",
        y="accuracy_mean",
    )
    file ="output/combined/plots/accuracy_vs_kernel.pdf"
    # Xlabels
    acc_graph1.set_xlabel("Metric")
    # rotate x labels
    for label in acc_graph1.get_xticklabels():
        label.set_rotation(45)
    # Ylabels
    acc_graph1.set_ylabel("Mean Accuracy (%)")

    # Recreate legend with only the two labels
    handles, labels = acc_graph1.get_legend_handles_labels()
    # remove the legend
    acc_graph1.legend_.remove()
    acc_graph1.legend(handles=handles, labels=labels, title="Kernel or Distance")
    # move legend outside of plot
    acc_graph1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # tight layout
    acc_graph1.figure.tight_layout()
    acc_graph1.figure.savefig(file)

    ########################################################################################
    # %%
    # def rbf_hamming_or_none(row):
    #     if "e^" in row['model.init.transform'] or "rbf" in row['model.init.transform']:
    #         row['RBF or Hamming'] = 'RBF'
    #     elif "hamming" in row['model.init.transform']:
    #         row['RBF or Hamming'] = 'hamming'
    #     else:
    #         row['RBF or Hamming'] = None
    #     return row

    # data = data.apply(rbf_hamming_or_none, axis=1)

    # %%
    sns.set_theme(context="paper", style="whitegrid", font="Times New Roman", font_scale=2)
    plt.gcf().clear()
    acc_graph2 = sns.barplot(
        data=data,
        hue='Hamming or RBF',
        x="Metric",
        y="accuracy_mean",
    )
    file ="output/combined/plots/accuracy_vs_kernel_function.pdf"
    # Xlabels
    acc_graph2.set_xlabel("Metric")
    # rotate x labels
    for label in acc_graph2.get_xticklabels():
        label.set_rotation(90)
    # Ylabels
    acc_graph2.set_ylabel("Mean Accuracy (%)")
    # Legend
    acc_graph2.legend(title="Algorithm")
    # move legend outside of plot
    acc_graph2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # tight layout
    acc_graph2.figure.tight_layout()
    acc_graph2.figure.savefig(file)



    # %%
    ########################################################################################
    subdata= data.copy()
    # Find the subset of the data that use kernels:
    mask = subdata['Kernel or Distance'] == "Kernel"
    subdata = subdata[mask]
    subdata['best_accuracy'] = subdata.groupby(['Dataset', 'Model'])['accuracy_mean'].transform('max')
    # Find the subset where accuracy_mean is equal to best_accuracy
    subdata = subdata[subdata['accuracy_mean'] == subdata['best_accuracy']]
    plt.gcf().clear()
    acc_graph3 = sns.barplot(
        data=subdata,
        hue='Model',
        x="Dataset",
        y="accuracy",
    )
    file ="output/combined/plots/accuracy_vs_model.pdf"
    # Xlabels
    acc_graph3.set_xlabel("Dataset")
    # rotate x labels
    for label in acc_graph3.get_xticklabels():
        label.set_rotation(90)
    # Ylabels
    acc_graph3.set_ylabel("5-Fold Accuracies \nfor Best Fit Model (%)")
    # Legend
    acc_graph3.legend(title="Algorithm")
    # move legend outside of plot
    acc_graph3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # tight layout
    acc_graph3.figure.tight_layout()
    acc_graph3.figure.savefig(file)

    # %%
    ########################################################################################
    subdata= data.copy()
    # Find the subset of the data that use kernels:
    subdata = subdata[subdata['Kernel or Distance'] == 'Kernel']
    subdata['best_accuracy'] = subdata.groupby(['Dataset', 'Model', 'algorithm'])['accuracy_mean'].transform('max')
    subdata = subdata[subdata['accuracy_mean'] == subdata['best_accuracy']]
    plt.gcf().clear()
    acc_graph4 = sns.barplot(
        data=subdata,
        hue='algorithm',
        x="Dataset",
        y="accuracy",
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"]
    )
    file ="output/combined/plots/accuracy_vs_algorithm.pdf"
    # Xlabels
    acc_graph4.set_xlabel("Dataset")
    # rotate x labels
    for label in acc_graph4.get_xticklabels():
        label.set_rotation(90)
    # Ylabels
    acc_graph4.set_ylabel("5-Fold Accuracies \nfor Best Fit Model (%)")
    # Legend
    acc_graph4.legend(title="Algorithm")
    # move legend outside of plot
    acc_graph4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    acc_graph4.figure.tight_layout()
    acc_graph4.figure.savefig(file)


    # %%
    ########################################################################################
    input_file = "output/combined/plots/precomputed_merged.csv"
    time_data = pd.read_csv(input_file, index_col=0)
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Times New Roman",
        font_scale=1.4,
    )
    plt.gcf().clear()
    distance_matrix_time_graph = sns.barplot(
        data=time_data,
        x="Metric",
        hue="algorithm",
        y="train_time_per_sample",
        order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
    )

    # Xlabels
    distance_matrix_time_graph.set_xlabel("Metric")
    # rotate x labels
    for label in distance_matrix_time_graph.get_xticklabels():
        label.set_rotation(90)
    # Ylabels
    distance_matrix_time_graph.set_ylabel("Distance Matrix Calculation \nTime per Sample (s)")
    # Legend
    distance_matrix_time_graph.legend(title="Algorithm")
    # move legend outside of plot
    distance_matrix_time_graph.legend()
    
    distance_matrix_time_graph.figure.set_size_inches(8, 3.5)
    # tight layout
    distance_matrix_time_graph.figure.tight_layout()
    distance_matrix_time_graph.figure.savefig(
        "output/combined/plots/distance_matrix_time_vs_algorithm.pdf",
    )
        


    # %%
    ########################################################################################
    input_file = "output/combined/plots/precomputed_merged.csv"
    time_data2 = pd.read_csv(input_file, index_col=0)
    time_data2 = time_data2[time_data2['data.sample.train_size'] == 1000]

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Times New Roman",
        font_scale=1.4,
    )
    plt.gcf().clear()
    pred_time_graph = sns.barplot(
        data=time_data2,
        x="Metric",
        hue="algorithm",
        y="predict_time_per_sample",
        # order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
    )
    # Xlabels
    pred_time_graph.set_xlabel("Metric")
    # rotate x labels
    for label in pred_time_graph.get_xticklabels():
        label.set_rotation(90)
    # Ylabels
    pred_time_graph.set_ylabel("Prediction Time per \n Sample (s)")
    # Legend
    pred_time_graph.legend(title="Algorithm")
    # move legend outside of plot
    pred_time_graph.legend()
    pred_time_graph.figure.set_size_inches(8, 3.5)
    
    # tight layout
    pred_time_graph.figure.tight_layout()
    pred_time_graph.figure.savefig("output/combined/plots/pred_time_vs_algorithm.pdf")



