This repository contains python code to process EEG data. It has been developped for a speed caping competition using a Biosemi EEG with 64 channels and thus contain some tweaks related to this context. Yet, we hope that these scripts provide a good overview of EEG signal processing for neuroergonomics using python. 

```
📦TutoEEG
 ┣ 📂Data    <-- EEG data should go there
 ┣ 📂Results <-- CSV results should go there
 ┣ 📂Scripts
 ┣ 📜GenericPipeline.pdf
 ┣ 📜README.md
 ┗ 📜setup.py
 ```

Data collected from speed caping subjects can be downloaded here : https://filesender.renater.fr/?s=download&token=0648bbea-13f7-4c4d-9aa8-4342e4397898

If the link is down please send a mail to : mathias.rihet@isae-supaero.fr

Three tasks have been performed by subjects :
- Resting State ('EO', 'EC' markers) where only EC should be considered in epochs from t=-0.5 to 25s
- Auditory Oddball ('normal', 'odd' markers) where both should be considered in epochs from t=-0.2 to 1s
- Calculation Task ('F pressed', 'D pressed' markers) with 'F' for easy condition and 'D' for difficult condition in epochs from t=-0.5 to 5s

A picture of both basic and advanced pipeline can be found in GenericPipeline.pdf
