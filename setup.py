from setuptools import setup

setup(name='SciRetreat',
      version='1.0',
      description='Basic EEG processing with python',
      author='Mathias Rihet',
      author_email='mathias.rihet@isae-supaero.fr',
    #   url='https://github.com/mathiasrihet/SciRetreat',
      install_requires = [
          'python == 3.11',
          'ipykernel == 6.29.4',
          'mne == 1.6.1',
          'seaborn == 0.13.2',
          'mne-icalabel == 0.6.0'
          'torch == 2.3.0',
          'meegkit == 0.1.7'
                          ],
)