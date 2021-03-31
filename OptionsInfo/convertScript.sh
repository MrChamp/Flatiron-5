#!/bin/bash
PATH=/home/steve/anaconda3/bin:/home/steve/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

jupyter nbconvert --to notebook --inplace --execute /home/steve/documents/flatIron/fIProject/OptionsInfo/optionDBGen.ipynb
