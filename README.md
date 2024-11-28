# Income Maps Using High-Resolution Satellite Imagery and Machine Learning

## Overview

In this repository, we provide the code and data for our study, where we explore the use of high-resolution satellite imagery and machine learning techniques to create detailed income maps. Specifically, we trained a convolutional neural network (CNN) using satellite images from the Metropolitan Area of Buenos Aires, Argentina, along with 2010 census data to estimate per capita income at a 50x50 meter resolution for the years 2013, 2018, and 2022. Our model, based on the EfficientnetV2 architecture, demonstrated a high accuracy in predicting household incomes, achieving an R2 score of 0.878, and surpassed the spatial resolution and performance of existing methods in the literature.

## Abstract

In this study, we examine the potential of using high-resolution satellite imagery and machine learning techniques to create income maps with a high level of geographic detail. We trained a convolutional neural network (CNN) with satellite images from the Metropolitan Area of Buenos Aires (Argentina) and 2010 census data to estimate per capita income at a 50x50 meter resolution for the years 2013, 2018, and 2022. This outperformed the resolution and frequency of available census information. The model, based on the EfficientnetV2 architecture, achieved high accuracy in predicting household incomes (R2 = 0.878), surpassing the spatial resolution and model performance of other methods used in the existing literature. This approach presents new opportunities for the generation of highly disaggregated data, enabling the assessment of public policies at a local scale, providing tools for better targeting of social programs, and reducing the information gap in areas where data is not collected.

## Repository Contents

- `scripts/`: All the scripts used are stored here. The main one is run_model.py, which uses functions from the other scripts. 
- `scripts/PyQGIS scripts`: The satellite imagery is processed using the scripts stored here.
- `README.md`: This file provides an overview and instructions for the repository.

## Relevant Links


- [Preprint Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5026760) - Deep Learning with Satellite Images Enables High-Resolution Income Estimation: A Case Study of Buenos Aires. R&R at PLOS ONE.
- [Visualize the income predictions!](https://ingresoamba.netlify.app) - In this link you can compare the satellite images of 2023 with the income predictions for 2013, 2018, and 2022.
- [Download the data](https://zenodo.org/records/13251268) - Predictions for AMBA region in geoparquet format and model weights at Zenodo repo.
  
### License

This project is licensed under the MIT License. See the LICENSE file for details.
