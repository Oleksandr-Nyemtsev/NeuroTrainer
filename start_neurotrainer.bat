@echo off

cd /d C:\Projects\NeuroTrainer

call C:\Users\user\anaconda3\Scripts\activate.bat neuro_env

python -m jupyter lab

pause