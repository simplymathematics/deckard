import os
import csv
from bs4 import BeautifulSoup


def generate_html_file(csv_file_path, output_folder):
    # Read the CSV file
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Get the title of the CSV file
    file_name = os.path.basename(csv_file_path)
    title = os.path.splitext(file_name)[0]

    # Create an HTML file path and open the file
    html_file_path = os.path.join(output_folder, f"{title}.html")
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
                if isinstance(cell, str) and os.path.exists(cell):
                    # Create a hyperlink with the capitalized name of the file
                    file_name = os.path.basename(cell)
                    link_title = os.path.splitext(file_name)[0]
                    cell = f'<a href="{cell}">{link_title.capitalize()}</a>'

                table_html += f"<td>{cell}</td>"
            table_html += "</tr>"
        table_html += "</table>"

        # Add the table to the HTML file
        soup.append(BeautifulSoup(table_html, "html.parser"))

        # Write the HTML content to the file
        html_file.write(soup.prettify())


def parse_folder(folder_path):
    # Create the output folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Iterate over the CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            csv_file_path = os.path.join(folder_path, file_name)
            generate_html_file(csv_file_path, folder_path)


# Define the folder path containing CSV files
folder_path = "output/reports"  # Update with your folder path

# Parse the folder and generate HTML files
parse_folder(folder_path)
