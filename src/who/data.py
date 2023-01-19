"""
    extra methods to prepare datasets before the training

    you can use 3 different datasets:
        - 'UCI HAR Dataset' by calling 'uci_har()'
        - 'Pedometer Project Dataset' by calling 'pedometer()'
        - 'WISDM Dataset' by calling 'wisdm()'

"""
import os
import shutil
import pandas as pd
import numpy as np
from subprocess import call
from os import listdir
from os.path import isdir, isfile, join

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_dir = os.path.join(root_dir, "src")
storage_path = os.path.join(root_dir, "storage")
dataset_path = os.path.join(storage_path, "dataset")
dataset_config_file = os.path.join(dataset_path, "dataset.conf")

def uci_har():
    """prepares 'UCI HAR Dataset' for training"""
    print("Dataset is preparing .....")
    source_path = os.path.join(storage_path, "UCI HAR", "UCI HAR Dataset")
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    call(['cp', '-r', source_path, storage_path])
    os.rename(os.path.join(storage_path, "UCI HAR Dataset"), dataset_path)
    
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]
    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]
    with open(dataset_config_file, "a+") as file:
        for label in LABELS:
            if label != LABELS[len(LABELS)-1]:
                file.write(label + ",")
            else:
                file.write(label + "\n")
    with open(dataset_config_file, "a+") as file:
        for signal in INPUT_SIGNAL_TYPES:
            if label != INPUT_SIGNAL_TYPES[len(INPUT_SIGNAL_TYPES)-1]:
                file.write(signal + ",")
            else:
                file.write(signal + "\n")
            

def pedometer():
    """prepares 'Pedometer Project Dataset' for training """
    x_data_file = os.path.join(dataset_path, "x.txt")
    y_data_file = os.path.join(dataset_path, "y.txt")

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)

    raw_dataset_dir = os.path.join(storage_path, "Pedometer")

    raw_dataset_sub_dirs = [f for f in listdir(raw_dataset_dir) if isdir(join(raw_dataset_dir, f))]
    dirs = [int(d.replace("P", "")) for d in raw_dataset_sub_dirs]

    raw_dataset_sub_dirs_sorted=[]
    raw_dataset_sub_dirs_copy = raw_dataset_sub_dirs
    while len(dirs)>0:
        smallest = min(dirs)
        smallest_index = dirs.index(smallest)
        raw_dataset_sub_dirs_sorted.append(raw_dataset_sub_dirs_copy[smallest_index])
        dirs.remove(smallest)
        raw_dataset_sub_dirs_copy.remove(raw_dataset_sub_dirs_copy[smallest_index])

    classes = ["Regular", "Irregular", "SemiRegular"]
    actions = []
    steps_data = []

    def get_position(position_str):
        if position_str == "Regular":
            return "1"
        if position_str == "Irregular":
            return "2"
        if position_str == "SemiRegular":
            return "3"

    x_size = 0
    y_size = 0

    for dir in raw_dataset_sub_dirs_sorted:
        sub_dir = os.path.join(raw_dataset_dir, dir)
        print(sub_dir)
        for cl in classes:
            data_file = os.path.join(sub_dir, cl, dir + "_"+cl+".txt")
            label_file = os.path.join(sub_dir,cl,"steps.txt")
            # print("/t"+data_file)
            # print("/t"+label_file)

            with open(label_file) as file:
                for row in file:
                    row = row.replace("\n", "")
                    row = row.split(" ")
                    steps_data.append(row)

            with open(data_file) as file:
                with open(x_data_file, "a+") as x:
                    with open(y_data_file, "a+") as y:
                        for index, value in enumerate(file):
                            move = False
                            for i in steps_data:
                                if int(i[0]) == index:
                                    y.write(i[1] + "\n")
                                    steps_data.remove(i)
                                    move = True
                                    break
                                
                            if move == False:
                                y.write("stand" + "\n")

                                

                            row = value.replace("\n", "")
                            x.write(row + " " + get_position(cl) + "\n")
                            x_size += 1
                            y_size += 1
                    
    print("Dataset is preparing .....")

    train_dir = os.path.join(dataset_path, "train")
    os.mkdir(train_dir)
    x_train_signals_dir = os.path.join(train_dir, "Inertial Signals")
    os.mkdir(x_train_signals_dir)
    x_wirst_acc_train = os.path.join(x_train_signals_dir, "wirst_acc_train.txt") 
    # x_wirst_acc_x_train = os.path.join(x_train_signals_dir, "wirst_acc_x_train.txt") 
    # x_wirst_acc_y_train = os.path.join(x_train_signals_dir, "wirst_acc_y_train.txt") 
    # x_wirst_acc_z_train = os.path.join(x_train_signals_dir, "wirst_acc_z_train.txt") 
    y_train_file = os.path.join(train_dir, "y_train.txt")

    test_dir = os.path.join(dataset_path, "test")
    os.mkdir(test_dir)
    x_test_signals_dir = os.path.join(test_dir, "Inertial Signals")
    os.mkdir(x_test_signals_dir)
    x_wirst_acc_test = os.path.join(x_test_signals_dir, "wirst_acc_test.txt") 
    # x_wirst_acc_x_test = os.path.join(x_test_signals_dir, "wirst_acc_x_test.txt") 
    # x_wirst_acc_y_test = os.path.join(x_test_signals_dir, "wirst_acc_y_test.txt") 
    # x_wirst_acc_z_test = os.path.join(x_test_signals_dir, "wirst_acc_z_test.txt") 
    y_test_file = os.path.join(test_dir, "y_test.txt")


    if (x_size == y_size):
        split_point = round(x_size*0.7)
    else:
        print("train data and label sizes are not equal")
        exit()

    def write_to_file(file_path, value):
        with open(file_path, "a+") as file:
            file.write(str(value))
            file.close()

    labels = []
    labels = ['stand', 'leftshift', 'rightshift', 'right', 'left'] 

    with open(x_data_file,"r") as x:
        with open(y_data_file,"r") as y:
            for index, x_data in enumerate(x):
                x_data = x_data.replace("\n","")
                x_data = x_data.split(" ")
                if index < split_point:
                    write_to_file(x_wirst_acc_train, str(x_data[0])+" "+str(x_data[1])+" "+str(x_data[2])+"\n")
                    # write_to_file(x_wirst_acc_x_train, str(x_data[0])+" ")
                    # write_to_file(x_wirst_acc_y_train, str(x_data[1])+" ")
                    # write_to_file(x_wirst_acc_z_train, str(x_data[2])+" ")
                else:
                    write_to_file(x_wirst_acc_test, str(x_data[0])+" "+str(x_data[1])+" "+str(x_data[2])+"\n")
                    # write_to_file(x_wirst_acc_x_test, str(x_data[0])+" ")
                    # write_to_file(x_wirst_acc_y_test, str(x_data[1])+" ")
                    # write_to_file(x_wirst_acc_z_test, str(x_data[2])+" ")
                
            for index, y_data in enumerate(y):
                y_data_label = y_data.replace("\n", "")
                # if y_data_label not in labels:
                #     labels.append(y_data_label)
                if index < split_point:
                    write_to_file(y_train_file, str(labels.index(y_data_label) + 1) + "\n")
                else:
                    write_to_file(y_test_file, str(labels.index(y_data_label) + 1) + "\n")
    
    INPUT_SIGNAL_TYPES = [
        # "wirst_acc_x_",
        # "wirst_acc_y_",
        # "wirst_acc_z_",
        "wirst_acc_"
    ]
    LABELS = [
        'stand', 
        'leftshift', 
        'rightshift', 
        'right', 
        'left'
    ] 
    with open(dataset_config_file, "a+") as file:
        for label in LABELS:
            if label != LABELS[len(LABELS)-1]:
                file.write(label + ",")
            else:
                file.write(label + "\n")
    with open(dataset_config_file, "a+") as file:
        for signal in INPUT_SIGNAL_TYPES:
            if label != INPUT_SIGNAL_TYPES[len(INPUT_SIGNAL_TYPES)-1]:
                file.write(signal + ",")
            else:
                file.write(signal + "\n")


def wisdm():
    """prepares 'WISDM Dataset' for training"""
    raw_dataset_file = os.path.join(storage_path, "WISDM_ar_v1.1_raw.txt")
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)

    LABELS = [
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing"
    ]
    INPUT_SIGNAL_TYPES = [
        "wirst_acc_",
        # "wirst_acc_x_",
        # "wirst_acc_y_",
        # "wirst_acc_z_",
    ]
    with open(dataset_config_file, "a+") as file:
        for label in LABELS:
            if label != LABELS[len(LABELS)-1]:
                file.write(label + ",")
            else:
                file.write(label + "\n")
    with open(dataset_config_file, "a+") as file:
        for signal in INPUT_SIGNAL_TYPES:
            if label != INPUT_SIGNAL_TYPES[len(INPUT_SIGNAL_TYPES)-1]:
                file.write(signal + ",")
            else:
                file.write(signal + "\n")

    train_dir = os.path.join(dataset_path, "train")
    x_train_signals_dir = os.path.join(train_dir, "Inertial Signals")
    x_train_acc = os.path.join(x_train_signals_dir, "wirst_acc_train.txt") 
    # x_train_acc_x = os.path.join(x_train_signals_dir, "wirst_acc_x_train.txt") 
    # x_train_acc_y = os.path.join(x_train_signals_dir, "wirst_acc_y_train.txt") 
    # x_train_acc_z = os.path.join(x_train_signals_dir, "wirst_acc_z_train.txt") 
    y_train_file = os.path.join(train_dir, "y_train.txt")

    test_dir = os.path.join(dataset_path, "test")
    x_test_signals_dir = os.path.join(test_dir, "Inertial Signals")
    x_test_acc = os.path.join(x_test_signals_dir, "wirst_acc_test.txt") 
    # x_test_acc_x = os.path.join(x_test_signals_dir, "wirst_acc_x_test.txt") 
    # x_test_acc_y = os.path.join(x_test_signals_dir, "wirst_acc_y_test.txt") 
    # x_test_acc_z = os.path.join(x_test_signals_dir, "wirst_acc_z_test.txt") 
    y_test_file = os.path.join(test_dir, "y_test.txt")

    os.mkdir(train_dir)
    os.mkdir(x_train_signals_dir)
    os.mkdir(test_dir)
    os.mkdir(x_test_signals_dir)

    columns = ["user_id", "activity", "timestamp", "acc_x", "acc_y", "acc_z"]
    data_list = []

    with open(raw_dataset_file) as raw_dataset:
        for raw_data in raw_dataset:
            raw_data = raw_data.replace(";","")
            raw_data = raw_data.replace("\n","")
            raw_data = raw_data.split(",")

            # print(raw_data)
            if(len(raw_data) == len(columns)):
                try:
                    if(abs(float(raw_data[5])!=0.0) or abs(float(raw_data[4])!=0.0) or abs(float(raw_data[3])!=0.0)):
                        data_list.append(raw_data)
                except:
                    pass

    df = pd.DataFrame(data=data_list, columns=columns)
    df = df.drop_duplicates(subset=["timestamp"])
    # duplicates = df[df.duplicated(subset=['timestamp'],keep="first")]
    # print(duplicates)
    # print(len(duplicates))

    data_n = df['activity'].value_counts().min()
    # print("count", data_n)

    Walking = df[df['activity'] == 'Walking'].head(data_n).copy()
    Jogging = df[df['activity'] == 'Jogging'].head(data_n).copy()
    Upstairs = df[df['activity'] == 'Upstairs'].head(data_n).copy()
    Downstairs = df[df['activity'] == 'Downstairs'].head(data_n).copy()
    Sitting = df[df['activity'] == 'Sitting'].head(data_n).copy()
    Standing = df[df['activity'] == 'Standing'].copy()

    # Walking = df.query('(activity == "Walking")').sample(n=data_n).copy()
    # Jogging = df.query('(activity == "Jogging")').sample(n=data_n).copy()
    # Upstairs = df.query('(activity == "Upstairs")').sample(n=data_n).copy()
    # Downstairs = df.query('(activity == "Downstairs")').sample(n=data_n).copy()
    # Sitting = df.query('(activity == "Sitting")').sample(n=data_n).copy()
    # Standing = df.query('(activity == "Standing")').sample(n=data_n).copy()

    balanced_df = pd.DataFrame()

    balanced_df = pd.concat([Walking,Jogging,Upstairs,Downstairs,Sitting,Standing])
    balanced_df["timestamp"] = balanced_df["timestamp"].astype('int')
    balanced_df = balanced_df.sort_values(by=['timestamp'], ascending=True)

    print("Dataset is preparing .....")

    for index, _ in balanced_df.iterrows():
        # old = balanced_df.at[index,'activity']
        balanced_df.at[index,'activity'] = LABELS.index(balanced_df.at[index,'activity']) + 1
        # print(old, "\t-\t", balanced_df.at[index,'activity'])

    split_point = round(data_n*0.7)

    def seperate_data(train_file_path, test_file_path, df_column, split_point):
        with open(train_file_path, "a+") as file:
            for i in df_column.loc[:split_point]:
                file.write(str(i)+"\n")
            file.close()
        with open(test_file_path, "a+") as file:
            for i in df_column.loc[split_point:]:
                file.write(str(i)+"\n")
            file.close()

    def seperate_data_2(train_file_path, test_file_path, df, split_point):
        df.drop(columns=["user_id", "activity"])
        with open(train_file_path, "a+") as file:
            for i in df.iloc[:split_point,:]:
                # file.write(str(i)+"\n")
                print(i)
            file.close()
        with open(test_file_path, "a+") as file:
            for i in df.iloc[split_point:,:]:
                # file.write(str(i)+"\n")
                pass
            file.close()


    seperate_data_2(x_train_acc, x_test_acc, balanced_df, split_point)
    # seperate_data(x_train_acc_x, x_test_acc_x, balanced_df["acc_x"], split_point)
    # seperate_data(x_train_acc_y, x_test_acc_y, balanced_df["acc_y"], split_point)
    # seperate_data(x_train_acc_z, x_test_acc_z, balanced_df["acc_z"], split_point)
    seperate_data(y_train_file, y_test_file, balanced_df["activity"], split_point)



