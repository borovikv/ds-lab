import os
import sys
import shutil


project_structure = [
    'data/predictions',  # The results of your model predictions.
    'data/processed',  # The final, canonical data sets for modeling.
    'data/raw',  # The original, immutable data dump.
    'models',  # Trained and serialized models, model predictions, or model summaries
    'notebooks',  # Jupyter notebooks.
    'reports',  # Generated analysis as HTML, PDF, LaTeX, etc.
    'srs',  # modules
]


def start_project(project_name):
    base_dir = os.path.dirname(__file__)
    for path in project_structure:
        os.makedirs(os.path.join(base_dir, 'projects', project_name, path))
    shutil.copy(os.path.join(base_dir, '.gitignore'), os.path.join(base_dir, 'projects', project_name, '.gitignore'))


if __name__ == '__main__':
    start_project(project_name=sys.argv[1])
