import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")


def cat_plot(
    data,
    x,
    y,
    hue,
    kind,
    titles,
    xlabels,
    ylabels,
    file,
    folder,
    legend_title=None,
    hue_order=None,
    rotation=0,
    set={},
    filetype=".eps",
    **kwargs,
):
    """
    The `cat_plot` function is a Python function that creates a categorical plot using seaborn library
    and saves it to a specified file in a specified folder.

    Args:
      data: The data parameter is the DataFrame that contains the data to be plotted. It should have
    columns corresponding to the x, y, and hue variables.
      x: The parameter "x" in the function "cat_plot" represents the variable that will be plotted on
    the x-axis of the categorical plot.
      y: The parameter "y" in the `cat_plot` function represents the variable that will be plotted on
    the y-axis of the categorical plot. It is the dependent variable or the variable of interest that
    you want to analyze or compare across different categories.
      hue: The "hue" parameter in the "cat_plot" function is used to specify the variable in the dataset
    that will be used to group the data points and create different colors for each group in the plot.
      kind: The "kind" parameter in the "cat_plot" function specifies the type of categorical plot to be
    created. It can take the following values. Check the seaborn.catplot documentation.
      titles: The `titles` parameter is a string or list of strings that specifies the titles of the
    subplots in the catplot. If it is a string, it will be used as the title for all subplots. If it is
    a list of strings, each string will be used as the title for
      xlabels: The `xlabels` parameter is used to set the label for the x-axis of the plot. It specifies
    the text that will be displayed as the label for the x-axis.
      ylabels: The `ylabels` parameter in the `cat_plot` function is used to set the label for the
    y-axis of the plot. It specifies the text that will be displayed as the label for the y-axis.
      file: The `file` parameter is the name of the file where the graph will be saved.
      folder: The "folder" parameter is the directory where the graph will be saved.
      legend_title: The `legend_title` parameter is used to set the title of the legend in the plot. If
    you want to provide a title for the legend, you can pass it as a string to the `legend_title`
    parameter when calling the `cat_plot` function.
      hue_order: The `hue_order` parameter is used to specify the order of the levels of the `hue`
    variable. It is a list that determines the order in which the different categories of the `hue`
    variable will be plotted.
      rotation: The "rotation" parameter in the "cat_plot" function is used to specify the rotation
    angle (in degrees) for the x-axis tick labels. By default, it is set to 0, which means the tick
    labels are not rotated. You can change the value of "rotation" to rotate. Defaults to 0
      set: The `set` parameter is a dictionary that allows you to set additional properties for the
    plot. You can pass any valid keyword arguments that are accepted by the `set()` method of the
    `seaborn.FacetGrid` object. These properties can be used to customize the appearance of the plot,
      filetype: The `filetype` parameter is used to specify the file extension for saving the graph. By
    default, it is set to ".eps", but you can change it to any other valid file extension such as
    ".png", ".jpg", etc. Defaults to .eps
    """

    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    logger.debug(
        f"Data sorted by x:{x}, y:{y}, hue:{hue}, kind:{kind}, hue_order:{hue_order}, and kwargs:{kwargs}.",
    )
    graph = sns.catplot(
        data=data, x=x, y=y, hue=hue, kind=kind, hue_order=hue_order, **kwargs
    )
    graph.set_xlabels(xlabels)
    graph.set_ylabels(ylabels)
    graph.set_titles(titles)
    if legend_title is not None:
        graph.legend.set_title(title=legend_title)
    else:
        graph.legend.remove()
    graph.set_xticklabels(graph.axes.flat[-1].get_xticklabels(), rotation=rotation)
    graph.set(**set)
    graph.tight_layout()
    graph.savefig(folder / file)
    plt.gcf().clear()
    logger.info(f"Saved graph to {folder / file}")


def line_plot(
    data,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    title,
    file,
    folder,
    y_scale=None,
    x_scale=None,
    legend={},
    hue_order=None,
    filetype=".eps",
    **kwargs,
):
    """
    The function `line_plot` is used to create a line plot with various customization options and save
    it to a specified file and folder.

    Args:
      data: The `data` parameter is the DataFrame that contains the data to be plotted.
      x: The parameter "x" is the name of the column in the dataset that will be used as the x-axis
    values in the line plot.
      y: The parameter `y` in the `line_plot` function represents the variable that will be plotted on
    the y-axis of the line plot. It is the dependent variable or the variable of interest that you want
    to visualize.
      hue: The "hue" parameter is used to specify a categorical variable that will be used to group the
    data points and differentiate the lines on the line plot. Each unique value of the "hue" variable
    will be represented by a different line on the plot.
      xlabel: The x-axis label for the line plot. It is the label that will be displayed on the x-axis
    of the graph.
      ylabel: The `ylabel` parameter is used to specify the label for the y-axis of the line plot. It is
    a string that represents the name or description of the data being plotted on the y-axis.
      title: The title of the line plot.
      file: The `file` parameter is the name of the file where the line plot will be saved. It should
    include the file extension.
      folder: The `folder` parameter is the directory where the generated graph will be saved.
      y_scale: The `y_scale` parameter is used to set the scale of the y-axis. It allows you to specify
    the type of scale to be used, such as "linear" for a linear scale, "log" for a logarithmic scale, or
    "symlog" for a symmetrical logarithmic
      x_scale: The `x_scale` parameter is used to set the scale of the x-axis. It can take the following
    values:
      legend: The `legend` parameter is a dictionary that allows you to customize the legend of the line
    plot. You can pass various options to the `legend` parameter to control the appearance of the
    legend. Some common options include:
      hue_order: The `hue_order` parameter is used to specify the order of the levels of the `hue`
    variable in the plot. It is a list that determines the order in which the different categories of
    the `hue` variable will be plotted.
      filetype: The `filetype` parameter specifies the file type of the saved graph. In the given code,
    the default value is set to ".eps", indicating that the graph will be saved as a PDF file. However,
    you can change the value of `filetype` to save the graph in a different. Defaults to .eps

    Returns:
      the line plot graph object.
    """
    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    graph = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, **kwargs)
    graph.legend(**legend)
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.set_title(title)
    if y_scale is not None:
        graph.set_yscale(y_scale)
    if x_scale is not None:
        graph.set_xscale(x_scale)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(folder / file)
    logger.info(f"Saved graph to {folder/file}")
    plt.gcf().clear()
    return graph


def scatter_plot(
    data,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    title,
    file,
    folder,
    y_scale=None,
    x_scale=None,
    legend={},
    hue_order=None,
    filetype=".eps",
    **kwargs,
):
    """
    The function `scatter_plot` creates a scatter plot using the provided data and parameters, and saves
    it to a specified file and folder.

    Args:
      data: The `data` parameter is the DataFrame that contains the data for the scatter plot.
      x: The parameter "x" in the scatter_plot function represents the variable that will be plotted on
    the x-axis of the scatter plot.
      y: The parameter "y" in the scatter_plot function represents the variable that will be plotted on
    the y-axis of the scatter plot.
      hue: The "hue" parameter in the scatter_plot function is used to specify a categorical variable
    that will be used to color the data points in the scatter plot. Each unique value of the "hue"
    variable will be assigned a different color in the plot.
      xlabel: The x-axis label for the scatter plot.
      ylabel: The `ylabel` parameter in the `scatter_plot` function is used to specify the label for the
    y-axis of the scatter plot. It is a string that represents the label you want to assign to the
    y-axis.
      title: The title parameter is used to specify the title of the scatter plot.
      file: The `file` parameter is the name of the file where the scatter plot will be saved.
      folder: The `folder` parameter is the directory where the scatter plot image will be saved.
      y_scale: The `y_scale` parameter is used to set the scale of the y-axis in the scatter plot. It
    can take values such as "linear" (default), "log", "symlog", "logit", etc. These values determine
    how the data is displayed on the y-axis. For
      x_scale: The `x_scale` parameter is used to set the scale of the x-axis. It can take values like
    "linear", "log", "symlog", "logit", etc. By default, if `x_scale` is not specified, the x-axis scale
    will be determined automatically based on
      legend: The `legend` parameter is a dictionary that allows you to customize the legend of the
    scatter plot. It can include the following keys:
      hue_order: The `hue_order` parameter is used to specify the order of the levels of the `hue`
    variable in the scatter plot. By default, the levels of the `hue` variable are ordered based on the
    order in which they appear in the data. However, if you want to specify a specific
      filetype: The `filetype` parameter is a string that specifies the file type of the saved graph. It
    is used to determine the file extension of the saved graph file. By default, it is set to ".eps",
    indicating that the graph will be saved as a PDF file. However, you can change. Defaults to .eps

    Returns:
      the scatter plot graph object.
    """

    plt.gcf().clear()
    file = Path(file).with_suffix(filetype)
    logger.info(f"Rendering graph {file}")
    data = data.sort_values(by=[hue, x, y])
    graph = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        **kwargs,
    )
    if y_scale is not None:
      graph.set_yscale(y_scale)
    if x_scale is not None:
      graph.set_xscale(x_scale)
    graph.set_xlabel(xlabel)
    graph.set_ylabel(ylabel)
    graph.legend(**legend)
    graph.set_title(title)
    graph.get_figure().tight_layout()
    graph.get_figure().savefig(Path(folder) / file)

    logger.info(f"Saved graph to {Path(folder) / file}")
    plt.gcf().clear()
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to the plot folder",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Data file to read from",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--plotfiletype",
        type=str,
        help="Filetype of the plots",
        default=".eps",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="INFO",
        help="Increase output verbosity",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file",
        default="conf/plots.yaml",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    assert Path(
        args.file,
    ).exists(), f"File {args.file} does not exist. Please specify a valid file using the -f flag."
    data = pd.read_csv(args.file)
    # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)

    if Path(args.path).absolute() == Path(args.path):
        logger.info("Absolute path specified")
        FOLDER = Path(args.path).absolute()
    else:
        logger.info("Relative path specified")
        FOLDER = Path(Path(), args.path)
    logger.info(f"Creating folder {FOLDER}")
    FOLDER.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to {FOLDER }")
    IMAGE_FILETYPE = (
        args.plotfiletype
        if args.plotfiletype.startswith(".")
        else f".{args.plotfiletype}"
    )
    if Path(FOLDER).exists():
        pass
    else:
        logger.info(f"Creating folder {FOLDER}")
        FOLDER.mkdir(parents=True, exist_ok=True)

    i = 0
    cat_plot_list = big_dict.get("cat_plot", [])
    for dict_ in cat_plot_list:
        i += 1
        cat_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)

    line_plot_list = big_dict.get("line_plot", [])
    for dict_ in line_plot_list:
        i += 1
        line_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)

    scatter_plot_list = big_dict.get("scatter_plot", [])
    for dict_ in scatter_plot_list:
        i += 1
        scatter_plot(data, **dict_, folder=FOLDER, filetype=IMAGE_FILETYPE)
