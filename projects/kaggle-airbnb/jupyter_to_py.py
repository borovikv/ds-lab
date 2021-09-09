import os
import glob
import subprocess


def convert_files():
    files = glob.glob("notebooks/*.ipynb")
    for i, file in enumerate(files):
        print(f'processing file {i+1}/{len(files)}:\n{file}')
        subprocess.run(["jupyter", "nbconvert", "--output-dir", "./src", "--to", "script", file])

        print('converting complete.')


if __name__ == '__main__':
    convert_files()
