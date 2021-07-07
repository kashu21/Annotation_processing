import os
import json
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from matplotlib import pyplot as plt
import statsmodels
from statsmodels.stats.inter_rater import fleiss_kappa

def image_url_converter(url: str):
    '''
    convert the image url into image number
    @param url: Url link for Image
    @return: Image number contained in the link
    '''
    return (os.path.splitext(os.path.basename(urlparse(url).path))[0])

def load_data(json_dataFile: str):
    '''
       return data
       @param json_dataFile: path to the file
       @return: List of python dictionaries object
       '''
    with open(json_dataFile, 'r') as datafile:  # load json data into python dictionary object
        loaded_data = json.load(datafile)

    with open('new_datafile.json', 'w') as f:  # just for better view/understanding of the data structure; not needed for the script
        json.dump(loaded_data, f, indent=2)

    new_dict_list = list()
    outer_dict_key_list = ["results", "root_node"]

    for node_id, inner_dict in loaded_data[outer_dict_key_list[0]][outer_dict_key_list[1]][outer_dict_key_list[0]].items():
        for task in inner_dict[outer_dict_key_list[0]]:
            new_dict_list.append(task)

    return new_dict_list

def convert_into_dataframe(data: list):
    '''
    return a dataframe created for the data
    :param data:  List of python dictionaries
    :return: return a dataframe
    '''
    dataframe = pd.json_normalize(data) #“Normalize” semi-structured JSON data into a flat table, i.e. dataframe

    dataframe.drop(columns = ['created_at','workpackage_total_size','project_root_node_input_id',
                   'user.id','user.vendor_id','project_node_input_id','project_node_output_id',
                   'root_input.image_url','loss'], inplace = True) # dropping the columns which are not required for the given tasks

    dataframe.rename(columns = {'task_input.image_url': 'image_num','task_output.answer': 'annotator_response',
                     'task_output.cant_solve': 'cant_solve','task_output.corrupt_data':'corrupt_data',
                     'task_output.duration_ms': 'annotation_duration_ms','user.vendor_user_id': 'annotator_id'}, inplace = True) #changing columns name

    dataframe['image_num'] = dataframe['image_num'].apply(image_url_converter) #converting image url to simply image number
    dataframe = dataframe[['image_num','annotator_id','annotator_response','cant_solve','corrupt_data', 'annotation_duration_ms']] # changing column order for better view of the data(optional)
    dataframe['annotator_response'] = dataframe['annotator_response'].replace({'no':False, 'yes':True}) #changing yes/no response to True/False
    dataframe.loc[(dataframe['corrupt_data']== True), 'annotator_response'] = "Corrupted_Image" #Including corrupt image response into annotator response column
    dataframe.loc[(dataframe['cant_solve']== True), 'annotator_response'] = "Undecided" #Including cant_solve response into annotator response column
    dataframe.drop(columns =['cant_solve', 'corrupt_data'], inplace = True) # dropping 'cant_solve' and 'corrupt_data' column after including their 'True' response
    # print(df.head(5))

    return dataframe

#######   TASKS   #######################################

def main():

    project_json_file_path = r"./anonymized_bicycle 1.1/anonymized_project.json"
    references_json_file_path = r"./anonymized_bicycle 1.1/references.json"

    if not os.path.exists("./plots"): #for saving plots
        os.makedirs("./plots")

    data_dict_list = load_data(project_json_file_path)
    df = convert_into_dataframe(data_dict_list)
    print("dataframe shape: ", df.shape)

    #How many annotators did contribute to the dataset?
    number_of_annotators = df['annotator_id'].nunique()
    print("\nNumber of annotators contributed: ",number_of_annotators)

    #What are the average, min and max annotation times?
    print("\n")
    print("Min,Max and Average annotation times\n", df['annotation_duration_ms'].describe().loc[['min','max', 'mean']])
    annotation_duration_barplot =df['annotation_duration_ms'].describe().loc[['min','max', 'mean']].plot.bar(y = ['min','max', 'mean'],figsize=(10, 10), color = 'r')
    annotation_duration_barplot.bar_label(annotation_duration_barplot.containers[0])
    plt.title("Annotation Duration(ms)")
    plt.show()
    annotation_duration_barplot.figure.savefig('./plots/Annotation_Duration_ms_plot.pdf')

    #Did all annotators produce the same amount of results, or are there differences?
    df.groupby(['annotator_id']).annotator_response.agg(['count'])
    number_of_responses_plot = df.groupby(['annotator_id']).annotator_response.agg(['count']).plot.bar(figsize=(10, 10))
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title(" Varying number of annotator Responses")
    plt.xlabel("Annotator_id")
    plt.ylabel("Number of responses")
    plt.show()
    number_of_responses_plot.figure.savefig('./plots/Varying_number_of_annotator_Responses_plot.pdf')


    #Are there questions for which annotators highly disagree?
    Inter_agreement_df = df.groupby(['image_num']).annotator_response.value_counts().unstack()
    Inter_agreement_df = Inter_agreement_df.fillna(0)
    IAA = fleiss_kappa(Inter_agreement_df, method='fleiss')
    print("\nInter_annotator_agreement: %.3f" %IAA)
    print("As Inter_annotator_agreement is within 0.81 – 1.00, so it is highly probable "
          "that there are no questions for which annotators highly disagree ")

    ######## kappa Interpretation ########## source : https://en.wikipedia.org/wiki/Fleiss%27_kappa
    # < 0	Poor agreement
    # 0.01 – 0.20	Slight agreement
    # 0.21 – 0.40	Fair agreement
    # 0.41 – 0.60	Moderate agreement
    # 0.61 – 0.80	Substantial agreement
    # 0.81 – 1.00	Almost perfect agreement


    #These are fields 'cant_solve' and 'corrupt_data' given in the task_output. How often does each occur in the project and do you see a trend within the
    # annotators that made use of these options?

    Uncertain_responses = df['annotator_id'].where((df['annotator_response'] == 'Undecided') | (df['annotator_response'] == 'Corrupted_Image'))
    Uncertain_responses.value_counts()
    Uncertain_responses_plot = Uncertain_responses.value_counts().plot.barh(figsize=(10, 10))
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Corrupt_data_Cant_solve responses")
    plt.show()
    Uncertain_responses_plot.figure.savefig('./plots/Corrupt_data_Cant_solve responses_plot.pdf')

    #Is the reference set balanced? Please demonstrate via numbers and visualizations.
    with open(references_json_file_path, 'r') as ref_file:
        ref_data = json.load(ref_file)

    ref_df = pd.DataFrame(ref_data).T
    ref_df.sort_index(inplace = True)
    dataset_content_ratio = ref_df.value_counts()
    print("\nContent proportion in dataset: ", dataset_content_ratio)
    dataset_content_ratio_plot = ref_df.value_counts().plot.bar(figsize=(10, 10))
    dataset_content_ratio_plot.bar_label(dataset_content_ratio_plot.containers[0])
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title("Balanced Dataset")
    plt.show()
    dataset_content_ratio_plot.figure.savefig('./plots/Balanced_Dataset_plot.pdf')


    #Using the reference set, can you identify good and bad annotators? Please use statistics and visualizations.
    comparison_df = df.pivot_table(index=['image_num'], columns='annotator_id', values=['annotator_response'], aggfunc='first')
    comparison_df.columns = comparison_df.columns.droplevel() #dropping 'annotator_reponse' column
    comparison_df = comparison_df.reset_index() # resetting index
    comparison_df.columns=comparison_df.columns.tolist() #converting to list object from series
    ref_df = ref_df.reset_index(drop = True) # resetting index before concatenating

    data_with_true_labels = pd.concat([comparison_df, ref_df], axis =1) #concateneating labels data into comparison_df
    data_with_true_labels = data_with_true_labels.set_index('image_num')

    bonafide_responses = {}
    for user in data_with_true_labels.columns[:-1]:
        match_counter = 0
        unmatch_counter = 0
        for index in data_with_true_labels.index:
            if not pd.isnull(data_with_true_labels.loc[index, user]):
                if data_with_true_labels.loc[index, user] == data_with_true_labels.loc[index, 'is_bicycle']:
                    match_counter += 1
                else:
                    unmatch_counter += 1

        bonafide_responses.update({user: {'Matched_response': match_counter, 'Unmatched_response': unmatch_counter}})

    bonafide_responses_df = pd.DataFrame(bonafide_responses).T
    bonafide_responses_plot = bonafide_responses_df.plot.bar(figsize=(10, 10))
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title("Annotations in comaprison with References dataset")
    plt.xlabel("Annotator_id")
    plt.ylabel("Number of responses")
    plt.show()
    bonafide_responses_plot.figure.savefig('./plots/Annotations_in_comaprison_with_References_dataset_plot.pdf')

    bonafide_responses_df['Total_number_of_responses'] = bonafide_responses_df.sum(axis = 1)
    bonafide_responses_df['Annotator_proficiency'] = (bonafide_responses_df['Matched_response']/ bonafide_responses_df['Total_number_of_responses'])
    Annotator_proficiency_plot = bonafide_responses_df.plot.bar(y = 'Annotator_proficiency',figsize=(10, 10), color = 'g')
    plt.xticks(rotation=30, horizontalalignment="right")
    plt.title("Annotator_Proficiency")
    plt.xlabel("Annotator_id")
    plt.ylabel("Proficiency score")
    plt.show()
    Annotator_proficiency_plot.figure.savefig('./plots/Annotator_proficiency_plot.pdf')


if __name__ == "__main__":
    main()





