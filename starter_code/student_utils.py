import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df["generic_drug_name"]=df["ndc_code"].apply(lambda x: ndc_df["Non-proprietary Name"][(ndc_df[ndc_df["NDC_Code"]==x].index.values[0])] if x is not np.nan else np.nan)

    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounters=pd.DataFrame(df.groupby(["patient_nbr"])["encounter_id"].head(1).values)
    first_encounter_df=df[df["encounter_id"].isin(first_encounters[0])]
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.copy()
    train=df.sample(frac=0.60,random_state=0)
    ts_val=df.drop(train.index)
    test=ts_val.sample(frac=0.5,random_state=0)
    validation=ts_val.drop(test.index)
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    high_cardinality_feat=["primary_diagnosis_code","other_diagnosis_codes","ndc_code"]
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column=tf.feature_column.categorical_column_with_vocabulary_file(c,
                                                                    vocabulary_file=vocab_file_path,
                                                                                num_oov_buckets=1)
        if(c in high_cardinality_feat):
            col=tf.feature_column.embedding_column(tf_categorical_feature_column,8)
#         elif(c =="age"):
#             #col=tf.feature_column.b(tf_categorical_feature_column,8)
        else:
            col=tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(col)
    return output_tf_list

#Question 8


def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/(std+10e-8)



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    
    '''
    new_list=["number_inpatient","number_emergency"]
    if(col in new_list):
        tf_numeric_feature=tf.feature_column.numeric_column(key=col,
                                                            default_value=default_value)
    else :
        normailzer_fx=functools.partial(normalize_numeric_with_zscore,mean=MEAN,std=STD)
        tf_numeric_feature=tf.feature_column.numeric_column(key=col,normalizer_fn=normailzer_fx,
                                                            default_value=default_value)
    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    for i, yhatt in enumerate(diabetes_yhat):
        mt = np.squeeze(yhatt.mean())
        st = np.squeeze(yhatt.stddev())
    return mt, st

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction=(col.flatten()>4).astype(int)
    return student_binary_prediction
