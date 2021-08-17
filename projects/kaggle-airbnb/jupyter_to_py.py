import os
import glob
import subprocess


def convert_files():
	files = glob.glob("notebooks/*.ipynb")
	for i, file in enumerate(files):
		print(f'processing file {i+1}/{len(files)}:\n{file}')
		subprocess.run(["jupyter", "nbconvert", "--to", "script", file])
		py_file = f'{file[:-5]}py'
		new_path = f"src{py_file[py_file.rfind('/'):]}"
		os.rename(py_file, new_path)
	print('converting complete.')


if __name__ == '__main__':
	convert_files()
