import csv
from pathlib import Path
from bs4 import BeautifulSoup
import argparse


def generate_html_file(csv_file_path, output_folder):
    # Read the CSV file
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Get the title of the CSV file
    file_name = Path(csv_file_path).name
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").title()
    # Create an HTML file path and open the file
    html_file_path = Path(output_folder, f"{title}.html")
    with open(html_file_path, "w") as html_file:
        # Create a BeautifulSoup object
        soup = BeautifulSoup("", "html.parser")
        # Add the title to the HTML file
        soup.append(BeautifulSoup(f"<h1>{title}</h1>", "html.parser"))
        # Create an HTML table from the CSV data
        table_html = "<table>"
        for row in data:
            table_html += "<tr>"
            for cell in row:
                # Check if the cell is a string representing a valid path
                if isinstance(cell, str) and Path(cell).exists():
                    # Create a hyperlink with the capitalized name of the file
                    file_name = Path(cell).name
                    link_title = (
                        Path(file_name).stem.replace("_", " ").replace("-", " ")
                    )
                    cell = f'<a href="{cell}">{link_title.capitalize()}</a>'

                table_html += f"<td>{cell}</td>"
            table_html += "</tr>"
        table_html += "</table>"

        # Add the table to the HTML file
        soup.append(BeautifulSoup(table_html, "html.parser"))

        # Write the HTML content to the file
        html_file.write(soup.prettify())


def main(folder_path, regex="*.csv"):
    # Create the output folder if it doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Iterate over the CSV files in the folder
    for file_name in Path(folder_path).glob(regex):
        if file_name.is_file():
            generate_html_file(file_name, folder_path)


parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, default="output/reports")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.folder_path)
