## PPG-affect-classification
This repository includes the code to process PPG signals using a lightweight pipeline outlined in the paper:
 J. Laiti, Y. Liu, P. J. Dunne, E. Byrne and T. Zhu, "Real-World Classification of Student Stress and Fatigue Using Wearable PPG Recordings," in IEEE Transactions on Affective Computing, doi: 10.1109/TAFFC.2025.3628467.

## Aim
The purpose of this project was to prompt further investigation into light-weight processing pipelines for affect detection. Specifically, these pipelines are intended to be efficient enough to run on wearable devices without cloud connectivity. Additionally, we were interested in exploring the potential differences in affect detection in adoelscent cohort data from real-world scenarios. 

## Datasets
This work focuses on evaluating a PPG preprocessing method and ML classification pipeline for three datasets:
- WESAD: A multimodal dataset of 15 adults wearing chest and wrist sensors (Empatica E4) during controlled baseline, amusement, and stress-inducing conditions in a laboratory setting.
- AKTIVES: A dataset of 25 children with neurdevelopmental disorders completing computer-based games with expert-labeled stress recognition
- Wellby: A custom dataset gathered in this research project based on adolescents self-reporting their stress and fatigue on a 5-point likert scale and completing one-minute PPG recordings during their daily life

## Repository Structure

This includes the following sections:

1. **Source code** in the `src` folder with:
   - Detailed processing pipeline functions in the `preprocessing` folder
   - Dataset-specific feature extraction scripts in the `feature_extraction` folder

2. **Example scripts** walking through the code used to generate the tables and figures in the manuscript

3. **Results tables** in the 'results' folder:
   - outcomes of the example scripts and tables presented in the manuscript within a folder for each dataset

4. **Compiled features data**:
   - outputs of the merged data from the open-source AKTIVES and WESAD datasets which can be recreated using the code in src/feature_extraction/wesad_td_ppg_extraction.py and src/feature_extraction/wesad_td_ppg_extraction.py. One WESAD dataset is was generated without SQI calculations and with bandpass of 0.7-10 while the other extracted SQI, used bp values of 0.5-10 and a data elimination threshold of 95 instead of 85%

## Getting Started

   1. Dataset access. While some summary features and results can be found in this repository, to run the examples you must download the open source datasets:
      - WESAD can be downloaded through [https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection]
      - AKTIVES must be requested through [https://www.synapse.org/Synapse:syn43685982.3/datasets/]
      - The Wellby dataset is not open-source due to privacy measures in this study.

   2. Set up the environment
      - This project was does in VSCode using python. There is a "requirments.txt" file with the necessary imports and "setup.py"

```bash
# Clone the repository
git clone https://github.com/j-laiti/PPG-affect-classification.git
cd PPG-affect-classification

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

   3. Explore! Work through any of the examples with the open access data or explore the functions provided and adapt them to your own work.

## Citation

If you use this code in your research, I'd be greatful if you cite:
```bibtex
@article{laiti2025realworld,
  title={Real-World Classification of Student Stress and Fatigue Using Wearable PPG Recordings},
  author={Laiti, Justin and Liu, Yu and Dunne, Padraic J. and Byrne, Elaine and Zhu, Tingting},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  doi={10.1109/TAFFC.2025.3628467}
}
```

## Contact
Thanks for making it this far! Feel free to reach out with any comments or questions and best of luck with your data processing endeavours :D