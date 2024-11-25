import pandas as pd
from tqdm import tqdm
import seaborn as sns


# Set seaborn theme to paper using times new roman font
sns.set_theme(context="paper", style="whitegrid", font="Times New Roman", font_scale=2)
if __name__ == "__main__":
    input_file = "output/combined/plots/search_merged.csv"
    output_file = "output/combined/plots/search_averaged.csv"
    val_cols = [
        "accuracy",
        "predict_time",
        "train_time",
        "train_time_per_sample",
        "predict_time_per_sample",
    ]
    do_not_group = [
        "model.init.distance_matrix_train",
        "model.init.distance_matrix_test",
    ]
    group_these = ["Dataset", "Model", "Metric", "algorithm", "model.init"]

    data = pd.read_csv(input_file, index_col=0)
    keep_these = group_these + val_cols
    tmp = []
    for col in keep_these:
        if col not in data.columns:
            for c in data.columns:
                if col in c:
                    tmp.append(c)
        else:
            tmp.append(col)
    data = data[tmp]

    # fillna with 0 because nans confuse the groupby
    data = data.fillna(0)
    print(f"Shape of data: {data.shape}")
    print(f"Columns: {data.columns}")
    data

    tmp_groups = []
    for col in group_these:
        if col not in data.columns:
            for c in data.columns:
                if col in c:
                    tmp_groups.append(c)
        else:
            tmp_groups.append(col)
    group_these = tmp_groups
    group_these = [col for col in group_these if col not in do_not_group]
    print(f"Grouping with {group_these}")
    grouped = data.groupby(group_these)
    # Calculate the mean and std for each group and each value column

    new_df = pd.DataFrame()
    for _, group in tqdm(grouped):
        for col in val_cols:
            # find subset from data
            subset = data.loc[group.index]
            # compute the mean
            mean = subset[col].mean()
            # compute the standard deviation
            std = subset[col].std()
            # add the mean and standard deviation to the group
            group[col + "_mean"] = mean
            group[col + "_std"] = std
            assert f"{col}_mean" in group.columns, f"{col}_mean not in group columns"
            assert f"{col}_std" in group.columns, f"{col}_std not in group columns"
            group = group.drop(col, axis=1)
            # group = group.head(1)
        new_df = pd.concat([new_df, group])
    new_df.to_csv(output_file)

    acc_graph = sns.catplot(
        data=new_df,
        x="Metric",
        row="Dataset",
        hue="algorithm",
        col="Model",
        y="accuracy_mean",
        kind="boxen",
        order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        row_order=["DDoS", "KDD NSL", "SMS Spam", "Truthseeker"],
        col_order=["KNN", "Logistic", "SVC"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
    )
    acc_graph.set_axis_labels("Metric", "Accuracy")
    acc_graph.set_titles("{row_name} - {col_name}")
    # Change legend title
    acc_graph._legend.set_title("Algorithm")
    # Rotate x labels
    for ax in acc_graph.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # tight layout
    acc_graph.tight_layout()
    acc_graph.savefig("output/combined/plots/accuracy_vs_algorithm.pdf")

    acc_graph = sns.catplot(
        data=new_df,
        hue="algorithm",
        row="Dataset",
        x="model.init.transform",
        col="Model",
        y="accuracy_mean",
        kind="boxen",
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
        row_order=["DDoS", "KDD NSL", "SMS Spam", "Truthseeker"],
        col_order=["KNN", "Logistic", "SVC"],
        sharex=False,
    )
    acc_graph.set_axis_labels("Kernel", "Accuracy")
    acc_graph.set_titles("{row_name} - {col_name}")
    # Change legend title
    acc_graph._legend.set_title("Algorithm")
    # Rotate x labels
    for ax in acc_graph.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # tight layout
    acc_graph.tight_layout()
    acc_graph.savefig("output/combined/plots/accuracy_vs_kernel.pdf")

    input_file = "output/combined/plots/precomputed_merged.csv"
    data = pd.read_csv(input_file, index_col=0)

    sns.set_theme(
        context="paper", style="whitegrid", font="Times New Roman", font_scale=1
    )
    distance_matrix_time_graph = sns.catplot(
        data=data,
        x="Metric",
        col="Dataset",
        hue="algorithm",
        y="train_time_per_sample",
        kind="boxen",
        order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        col_order=["DDoS", "KDD NSL", "SMS Spam", "Truthseeker"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
    )
    distance_matrix_time_graph.set_axis_labels(
        "Metric", "Distance Matrix Calculation \n Time per Sample (seconds)"
    )
    distance_matrix_time_graph.set_titles("{col_name}")
    # Change legend title
    distance_matrix_time_graph._legend.set_title("Algorithm")
    # Rotate x labels
    for ax in distance_matrix_time_graph.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # Take up 1/3 of an A4 page
    distance_matrix_time_graph.figure.set_size_inches(8, 3.5)
    # tight layout
    distance_matrix_time_graph.tight_layout()
    distance_matrix_time_graph.savefig(
        "output/combined/plots/distance_matrix_time_vs_algorithm.pdf"
    )
    sns.set_theme(
        context="paper", style="whitegrid", font="Times New Roman", font_scale=2
    )
    train_time_graph = sns.catplot(
        data=new_df,
        x="Metric",
        row="Dataset",
        hue="algorithm",
        y="train_time_per_sample_mean",
        kind="boxen",
        order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        row_order=["DDoS", "KDD NSL", "SMS Spam", "Truthseeker"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
        col="Model",
        col_order=["KNN", "Logistic", "SVC"],
    )
    train_time_graph.set_axis_labels(
        "Metric", "Model Training Time \n per Sample (seconds)"
    )
    train_time_graph.set_titles("{row_name} - {col_name}")
    # Change legend title
    train_time_graph._legend.set_title("Algorithm")
    # Rotate x labels
    for ax in train_time_graph.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # Take up 2/3 of an A4 page
    train_time_graph.figure.set_size_inches(8, 8)
    train_time_graph.tight_layout()
    train_time_graph.savefig("output/combined/plots/train_time_vs_algorithm.pdf")

    sns.set_theme(
        context="paper", style="whitegrid", font="Times New Roman", font_scale=1.5
    )
    pred_time_graph = sns.catplot(
        data=new_df,
        x="Metric",
        row="Dataset",
        hue="algorithm",
        y="predict_time_per_sample_mean",
        kind="boxen",
        order=["GZIP", "BZ2", "Brotli", "Hamming", "Ratio", "Levenshtein"],
        row_order=["DDoS", "KDD NSL", "SMS Spam", "Truthseeker"],
        hue_order=["Vanilla", "Assumed", "Enforced", "Average"],
        col="Model",
        col_order=["KNN", "Logistic", "SVC"],
    )
    pred_time_graph.set_axis_labels("Metric", "Prediction Time per Sample (seconds)")
    pred_time_graph.set_titles("{col_name} - {row_name}")
    # Change legend title
    pred_time_graph._legend.set_title("Algorithm")
    # Rotate x labels
    for ax in pred_time_graph.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # tight layout
    pred_time_graph.tight_layout()
    pred_time_graph.savefig("output/combined/plots/pred_time_vs_algorithm.pdf")
