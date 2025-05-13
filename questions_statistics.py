import csv
import json
import os
import re
from collections import defaultdict
import time
from pathlib import Path

from question_generator import generate_questions_list, get_l3_questions, get_reports_with_positive_l1_questions, Mode, \
    get_l1l2l3_questions

DATA_DIR_TRAIN = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\train_vectorized_answers")
DATA_DIR_VAL = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\val_vectorized_answers")
DATA_DIR_TEST = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\test_vectorized_answers")

DATA_DIRS = [DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST]


#report_files = sorted(os.listdir(DATA_DIR_QA), key=lambda x: int(x.split(".")[0]))

questions_count = {}
#questions_count_final = {}
sanity_count = 0


def negative_response(path):
    no = "_no"
    if no in path:
        start_index = path.index(no)
        tripped_str = path[start_index+3:]
        # if _no is not at the end, happens with _nodjghdjfg_no_1
        if no in tripped_str:
            start_index = tripped_str.index(no)
            tripped_str = tripped_str[start_index + 3:]
            len_after_no = len(tripped_str)
            if len_after_no <= 3:
                return True
        len_after_no = len(tripped_str)
        if len_after_no <= 3:
            return True

def is_l1_question(path):
    if path.endswith("_yes"):
        return True

def is_l2_question(path):
    # in body_regions_something_yes_0 that would be the s_0 underscore
    last_underscore_index = path.rfind("_")
    # in body_regions_something_yes_0 that would be the g_y underscore
    second_to_last_underscore_index = path.rfind("_", 0, last_underscore_index)
    last_path = path[second_to_last_underscore_index:last_underscore_index]

    if "yes" in last_path:
        return True

    return False

def is_instance(path):
    if "1" in path or "2" in path or "3" in path or "4" in path:
        return True

# sum different instances of the same paths
# abdoment_sign_yes_0 and abdoment_sign_yes_1/2/3
# all their values summed in abdoment_sign_yes
def summed_instances(summed_reports):
    merged_instances = defaultdict(int)

    # find occurances of _0, _1, _2 ...
    pattern = re.compile(r"_(\d+)$")

    # remove instance counter from paths
    # abdoment_sign_yes_0/1/2/3 to abdoment_sign_yes
    for path, occurances in summed_reports.items():
        new_path = pattern.sub("", path)
        merged_instances[new_path] += occurances

    merged_instances = dict(merged_instances)

    return merged_instances


#Generate the pathologies_occurances.json
# Go through all report files, get all non negative responses - paths that don't have _no
# and sum how many times they occur

def vectorized_summation():
    all_reports = []
    for data_dir in DATA_DIRS:
        report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split(".")[0]))
        for report_file in report_files:
            with (open(data_dir / report_file) as json_report):
                report_dict = json.load(json_report)
                all_reports.append(report_dict)



        # Replace the true/false in values with 1/0
        # reports_int = []
        # for report in all_reports:
        #     report_int = {}
        #     for key,value in report.items():
        #         report_int[key] = int(value)
        #     reports_int.append(report_int)

        summed_reports_vector = all_reports[0]
        for report in all_reports[1:]:
            for key,value in report.items():

                summed_reports_vector[key] = int(summed_reports_vector[key]) + int(value)

    # Same as above but done in pythonic fashion
    #vectors = [[int(value) for value in report.values()]for report in all_reports]
    #summed_vector = [sum(x) for x in zip(*vectors)]
    #keys = all_reports[0].keys()

    #summed_reports = {key: sum(int(report[key]) for report in all_reports) for key in keys}


    # print(f"Time taken for manual: {end_time_manual-start_time_manual} seconds")
    # print(f"Time taken for pythonic: {end_time_pythonic-start_time_pythonic} seconds")

    # with open("vectorized_summation_nol1.json", "w") as outfile:
    #     json.dump(summed_reports, outfile)

    print(f"Vectorized summation done!")
    return summed_reports_vector



# generate a list of dicts of the form
# {path: {count: 4
#          files: [1, 6, 5, 7]}
# paths is the path of a question
# count is how many times that path occurs across
# all reports (train, val and test)
# files is the files where that path occurs and is given as true
def generate_reports_answers_files_():
    all_reports = []
    reports_total = {}
    for data_dir in DATA_DIRS:
        report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split(".")[0]))
        for report_file in report_files:
            with (open(data_dir / report_file) as json_report):
                report_dict = json.load(json_report)

                for path,value in report_dict.items():

                    if negative_response(path) or is_instance(path):
                        continue
                    else:
                        # if that path is not yet added in the sum
                        if int(value) > 0:
                            if path not in reports_total:
                                reports_total[path] = [report_file]
                            else:
                                reports_total[path].append(report_file)
                        else:
                            if path not in reports_total:
                                reports_total[path] = []

            print(f"Processed file: {report_file}!")

            #all_reports.append(report_dict)


    # summed_reports_vector = all_reports[0]
    # for report in all_reports[1:]:
    #     for key,value in report.items():
    #
    #         summed_reports_vector[key] = int(summed_reports_vector[key]) + int(value)


    with open("detailed_occurance_summary_trainvaltest.json", "w") as outfile:
        json.dump(reports_total, outfile)

    print(f"Vectorized summation done!")
    return reports_total


# get the L3 questions dict
def filter_summed_vectors(summed_reports):
    positive_report_questions = {}
    for path in summed_reports:
        # exclude negative responses
        # exclude l1 questions
        # need to exclude l2 questions yes_0/1/2/3/4 but make an exception
        # for paths that contain infos
        if not negative_response(path) and not is_l1_question(path) and not is_l2_question(path):
            positive_report_questions[path] = summed_reports[path]


    return positive_report_questions

def remove_instances_from_path(reports_summaries):
    # remove the _0 , _1, _2 at the end of the path
    removed_instances = {}
    for path in reports_summaries:
        # exclude negative responses
        # exclude l1 questions
        # need to exclude l2 questions yes_0/1/2/3/4 but make an exception
        # for paths that contain infos
        if not negative_response(path):
            split_path = path.split("_")

            if split_path[-1].isdigit():
                split_path = split_path[:-1]
            split_path = '_'.join(split_path)
            removed_instances[split_path] = reports_summaries[path]

    return removed_instances


reports_summaries = generate_reports_answers_files_()
#filtered_reports = filter_summed_vectors(reports_summaries)
filtered_reports = remove_instances_from_path(reports_summaries)
#no_instances = remove_instances_from_path(reports_summaries)

#questions_count_final = vectorized_summation()
#filtered_reports = filter_summed_vectors(questions_count_final)



#print(f"File : {report_file} processed")
print(f"All done!")
print(f"Sanity count: {sanity_count}")
#Write the statistics to a file

# l1_reports_qa = get_reports_with_positive_l1_questions(Mode.QA_PAIRS)
# l1_reports_vec = get_reports_with_positive_l1_questions(Mode.VECTORIZED_ANSWERS)
#
# qa_set = set(l1_reports_qa)
# vec_set = set(l1_reports_vec)
#
# sym_diff = qa_set.symmetric_difference(vec_set)
#
# print(len(l1_reports_qa))
# print(len(l1_reports_vec))
#
# question = generate_questions_list("report_questions_reference_infos.csv")


# with open("pathologies_occurances8.json", "w") as json_file:
#     json.dump(filtered_reports, json_file)
#
# with (open(DATA_DIR_QA / "2.json") as json_report):
#     report_dict = json.load(json_report)
#
# not_found = []
# for report_entry in report_dict:
#     for occurance in filtered_reports:
#         if occurance in report_entry:
#             break
#     else:
#         not_found.append(report_entry)
#
# with open('missing_summed_instances.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     for missing in not_found:
#         writer.writerow([missing])



l1_questions = generate_questions_list("report_questions_reference_infos.csv")
# Format: key: [L3 question, L2 question, L1 question]
#l3_questions_map = get_l3_questions(l1_questions)
l1l2l3_questions_map = get_l1l2l3_questions(l1_questions)

matches = 0
#summed_instances_reports = summed_instances(filtered_reports)
with open('detailed_occurance_summary_trainvaltest5.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    for l3_question in filtered_reports:
        ## comment out after
        split_l3_path = l3_question.split("_")
        #if split_l3_path[-1]=='yes':
        split_l3_path = split_l3_path[:-1]
        path_without_answer = '_'.join(split_l3_path)
        # comment out after
        for l3_stat in l1l2l3_questions_map:
            if l3_stat == path_without_answer:
                matches += 1
                last_index = len(l3_stat)-1
                remaining_part = l3_question[last_index:]
                # brokeback way
                # i1 = remaining_part.index("_")
                # remaining_part = remaining_part[i1+1:]s
                # i1 = remaining_part.index("_")
                # remaining_part = remaining_part[:i1]
                pattern = r"[a-z ]{2,}"
                match = re.findall(pattern, remaining_part)

                path = l3_stat
                questions_to_path = l1l2l3_questions_map[l3_stat]
                answer = match[0]
                answer_num_occurances = filtered_reports[l3_question]

                writer.writerow([l3_question, questions_to_path, answer, len(answer_num_occurances), answer_num_occurances])





print(f"Number of matches in the dicts: {matches}")
print(f"Number entries in the questions_map: {len(l1l2l3_questions_map)}")
print(f"Number entries in the occurances: {len(filtered_reports)}")
