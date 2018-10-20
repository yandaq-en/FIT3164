This project implements the ICU mortality prediction model using XBGoost. Several instructions are provided below to help users to understand the project implementation.

  1. Hyperopt, xgboost and shap libraries are used for this project, please ensure these packages have been installed before runing the project code. To help users install packages, a list of packages needed is provided in the requirements.txt file. After download the project folder, run command line in the folder and use "pip install -r requirements.txt" command to install packages needed for this project. Notice, xgboost may not be able to install directly for windows users, as xgboost is no longer supported for pip install. Besides, C++ tools are needed to install shap. Please ensure C++ tools are installed before running the command.
  
  
  2. In order to observe the individual explanation, one have to use Juypter notebook, as the ploting method uses Javascript which is not avaliable in Pycharm.
  
  3. An applicaition.py file is provided as a usable application. Users can enter the value of features and predict the probability of surviving. param.txt, selected_train_X.csv, train_y.csv are files needed for running the application. These files can be generated from the main.py file.
  
  4. The main.py file is the the main code for this project, including data preprocessing, modelling and explainer implementation. Detailed documentation is provided in the file. In contrast, the application.py file has no documentation inside as it is just the application code and it is not the project interest.
  
  5. A Juypter notbook version of main.py is also provided in the folder. This is for the convinence of users who may use juypter notebook to read python files. Also, users can use this file to observe the individual explanation.
  
  6. In order to use the main.py code by default, three folder of datasets have to be in the same folder with the main.py code. This means the main.py code can call "set-a/" and find the folder. Alternatively, users can change the path of the directory to fit their constom use.
  
  7. The github link for this project is provided here: https://github.com/YandaQu/FIT3164
  
  8. The datasets used in this proejct can be downloaded from Physionet.com, the URL links to the data source is provided below:
    https://physionet.org/challenge/2012/
