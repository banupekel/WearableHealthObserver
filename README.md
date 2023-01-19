# WearableHealthObserver
repo for  wearable health observer

---

AI development for 'Human Healt Observer'

--- 

## Prepare 
### **- MacOS Apple Slicon**

1. Install homebrew
2. Install miniconda
3. Install Xcode Command Line Tools (`$ xcode-select --install`)
4. Follow the instructions below :    
```
    # turn off the default base env
    $ conda config --set auto_activate_base false

    # create env and activate it
    $ conda create --name pekel python=3.8
    $ conda activate pekel

    # Install the Tensorflow dependencies:
    $ conda install -c apple tensorflow-deps

    # Install base TensorFlow:
    $ pip install tensorflow-macos

    # Install metal plugin
    $ pip install tensorflow-metal

    # Install other requirements
    $ pip install -r requirements.txt

    # verify the installation
    $ python ./src/test.py
```

### **- Other OSes With Intel Chip**
1. Install miniconda
2. Follow the instructions below :    
```
    # turn off the default base env
    $ conda config --set auto_activate_base false

    # create env and activate it
    $ conda create --name pekel python=3.8
    $ conda activate pekel

    # Install other requirements
    $ pip install -r requirements.txt

    # verify the installation
    $ python ./src/test.py
```

### **- Download the dataset**

- "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
- "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.names"
- "https://cecas.clemson.edu/tracking/Pedometer/Data.zip"
  
```
    # Linux and MacOS
    $ cd ./storage

    $ wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
    $ unzip -nq "UCI HAR Dataset.zip"  -d "UCI HAR"

    $ wget "https://cecas.clemson.edu/tracking/Pedometer/Data.zip"
    $ unzip -nq "Data.zip"  -d "Pedometer"



    # Windows
    $ python -m wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
    $ python -m unzip -nq "UCI HAR Dataset.zip"
```

---

## Run
```
    $ python ./src/main.py
```