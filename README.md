# Flatiron-5
Project 5 for Flatiron Bootcamp DS Program

# Bitcoin Weighted Price Forecast
<p>This project is focused on engineering a database for financial data (specifically historical stock and options data).

There is also some cursory analysis of the data, and an attempt at an oversimplified GAN model</p>

## Contents
 <ul><b>- DF </b></br>
      <p>- The files for generating a dataFrame from the database</p></ul>
 <ul><b>- GAN </b></br>
      <p>- The files for the GAN model</p></ul>
 <ul><b>- MySQL </b></br>
      <p>- ER diagram as well as MySQL-workbench backup files</br>
         - A program for adding feature columns to the database</p></ul>
 <ul><b>- OptionsInfo </b></br>
      <p>- The files for creating a crontab job to run the options collection program</br>
         - The Options historic data collection program</p></ul>
 <ul><b>- Stocks</b></br>
      <p>- The files for adding the .csv stock data into the database</p></ul>
 <ul><b>- resourcesAndTutorials</b></br>
      <p>- Various tutorials, templates, and resources used for building some files</p></ul>
 <ul><b>- sa</b></br>
      <p>- The sentiment analysis folder</br>
         - Contains the framework for future sentiment analysis programs and files</p></ul>


## Getting Started
<p>The files are in the .ipynb format and were made in Jupyter Lab using the dark theme; so color schemes may be difficult to see on lighter themed environments.</br></br>

The "optionDBGen.ipynb" file is the focus of the project - it was written with the intention of collecting, cleaning and organizing historic options data to be stored in a database for future analysis. The accompanying cronJob file should be run in a linux environment as a cronTab input in order to automate the collection process.</br></br>

The "simpleTSGAN.ipynb" file contains the rudimentary GAN model that will be improved in the future.
</p>

## Prerequisites
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
 - Built with Jupyter Lab version 2.2.6
## Installing
<p>The majority of the files can be downloaded and run locally with Jupyter Lab or Jupyter Notebook</br></br>

These files were intended to be run on a linux system; the host system is Ubuntu v 20.04
</p>

## Writeup
https://
## Authors
- MrChamp
