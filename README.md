# LNC-SCOPE: LncRNA Structural & miRNA-Interaction Pipeline

**Authors:** Kasukurthi Ananya & Parnika Pawar

---

## Project Overview
LNC-SCOPE is a Python-based pipeline developed to analyze the regulatory potential of long non-coding RNA (lncRNA). This tool provides an integrated approach to RNA analysis by combining custom secondary structure folding with miRNA binding site prediction to identify functional "hotspots" and classify the RNA's biological role, such as a miRNA sponge.

## Primary Objective
The primary objective of LNC-SCOPE is to develop a computational framework that bridges the gap between RNA structural biology and functional genomics. Specifically, this tool aims to:
* Predict the secondary structure of lncRNA using thermodynamic energy models.
* Identify physically accessible regions within the RNA fold to determine binding availability.
* Map high-confidence miRNA interactions to characterize the lncRNA’s regulatory role (e.g., as a miRNA sponge).

## Files in this Submission
* **`lnc-scope.py`**: The main Python script containing the full analysis pipeline.
* **`mirna_sequences.csv`**: The required database of miRNA sequences used for interaction mapping.
* **`examples folder`**: Example input files in the format of FASTA and also dot-bracket structure format text files
* **`README.md`**: This documentation file providing project context and instructions.
* **`usp.txt`**: The documentation file providing the Unique Selling Proposition of this project.

## How to Run the Program
To ensure the program runs correctly, please follow these steps:

1.  **Requirements**: The program is written in Python 3 and uses standard libraries (`json`, `time`, `csv`, `os`). No external bioinformatics suites (like ViennaRNA) are required.
2.  **Environment**: Place `lnc-scope.py`, `lncRNA.fasta`(input file) and `mirna_sequences.csv` in the same directory.
3.  **Execution**: Open a terminal or IDE and run:
    ```bash
    python lnc-scope.py
    ```
4.  **Input**:
    * The program will ask if you have a **dot-bracket structure**. If "yes," you can **manually paste** the dot-bracket structure directly into the prompt.
    * If "no," yhe program will ask if you have a **FASTA file**. If "yes," provide the file path.
    * If "no," you can **manually paste** the RNA or DNA sequence directly into the prompt.
5.  **Outputs**: Upon completion, the script generates two JSON files in the local directory:
    * `structure_output.json`: Intermediate data regarding folding and accessibility.
    * `final_lnc_scope_output.json`: The final interaction results and functional classification.



## Key Features
* **Custom RNA Folding**: We implemented a Dynamic Programming (DP) algorithm that calculates secondary structures using an energy model accounting for base-pair energy ($G-C$, $A-U$, $G-U$) and stacking bonuses.
* **Accessibility & Zone Classification**: The pipeline identifies 6-mer regions likely to be available for binding and categorizes them as Sponge Zones, Scaffold Candidates, or Hybrid Zones.
* **miRNA Interaction Stage**: Includes a pre-filtering step using 6-mer seed matching for efficiency, followed by duplex energy calculations ($\Delta G$) to identify high-confidence binding sites.
* **Mutation Sensitivity**: A diagnostic scan that identifies specific nucleotides where a mutation would significantly disrupt the predicted secondary structure.

## Authorship
This project was collaboratively designed, implemented, and tested by **Parnika Pawar** and **Kasukurthi Ananya**.