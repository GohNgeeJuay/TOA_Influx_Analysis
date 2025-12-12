# Solar Data Processing and Analysis (Top-of-atmosphere Irradiation)

### Author: Jake Goh
### Date: 11th December 2025


This repository contains data, Python scripts, and notebooks for processing and analyzing solar irradiance data. It supports downloading, cleaning, and visualizing datasets for renewable energy research and related sustainability projects.

The code complements the blog: ...

---

### Repository Structure

toa_influx_analysis/

├── data (this is where the data is stored)

├── notebooks (the .ipynb Exploratory Data Analysis files are stored here)

├── src (this is where the functions written for TOA influx are stored)

├── tests (python code to run the tests on the TOA influx functions in the src folder)

├── environment.yml (list of packages needed for this project)


To reproduce the findings in the notebooks:

## Set up
1. Clone the repository
   `git clone https://github.com/GohNgeeJuay/TOA_Influx_Analysis.git`
2. Create conda environment:
   ```
   conda env create -f environment.yaml
   conda activate toa-influx
   ```
3. To replicate the findings, run the `notebooks\toa_influx.ipynb`
4. To perform tests created in the `tests\`, go to that directory and run `pytest`.


## Contact
GitHub: https://github.com/gohngeejuay
Linkedin: www.linkedin.com/in/gohngeejuay

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







