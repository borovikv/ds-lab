import os
import sys

project_structure = [
    'data/original',  # The original, immutable data dump.
    'data/processed',  # The final, canonical data sets for modeling.
    'data/results',  # The results of your project work.
    'models',  # Trained and serialized models, model predictions, or model summaries.
    'notebooks',  # Jupyter notebooks.
    'reports',  # Generated analysis as HTML, PDF, LaTeX, etc.
    'sql',  # queries.
]


def start_project(project_name):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for path in project_structure:
        os.makedirs(os.path.join(base_dir, 'jupiter-projects', project_name, path))


if __name__ == '__main__':
    start_project(project_name=sys.argv[1])
