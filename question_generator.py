import json
import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
import csv


from data_utils.preprocessing_radrestruct import get_topic_question, get_question
from evaluation.defs import YES, NO, NOT_PREDICTED
from evaluation.predict_autoregressive_VQA_radrestruct import get_value
from question import Question

DATA_DIR_NEW_REPORTS = "E:\Development\ad_Rad-ReStruct\data\radrestruct\new_reports"

#DATA_DIR_VECTORIZED_ANSWERS = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\train_vectorized_answers")

DATA_DIR_TRAIN = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\train_vectorized_answers")
DATA_DIR_VAL = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\val_vectorized_answers")
DATA_DIR_TEST = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\test_vectorized_answers")

DATA_DIR_TRAIN_QA = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\train_qa_pairs")
DATA_DIR_VAL_QA = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\val_qa_pairs")
DATA_DIR_TEST_QA = Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\test_qa_pairs")

DATA_DIRS_QAPAIRS = [DATA_DIR_TRAIN_QA, DATA_DIR_VAL_QA, DATA_DIR_TEST]
DATA_DIRS_VECTORIZED_ANSWERS = [DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST]


REPORT_QUESTIONS_REFERENCE = Path("E:\\Development\\ad_Rad-ReStruct\\'report_questions_reference.csv'")

history_ad = {}
history_q = []
preds = []
pred_vector = []
match_instances = []
report_vector_gt = []

def iterate_instances_VQA(elem, question, elem_name, topic_name, area_name, history, pred_vector, report_keys,
                          max_instances, answer_options, report_vector_gt=None, match_instances=False):
    infos = elem["infos"] if topic_name != "infos" else elem
    elem_history = deepcopy(history)

    gt_instances = elem["instances"]
    instance_keys = list(infos.keys())
    if "instances" in instance_keys:
        instance_keys.remove("instances")
    # make sure all instances have same structures
    for i in range(len(gt_instances)):
        assert instance_keys == list(gt_instances[i].keys())
    no_predicted = False

    max_instance_key = f"{area_name}/{topic_name}" if topic_name == 'infos' else f"{area_name}/{elem_name}"
    max_num_occurences = max_instances[max_instance_key] if max_instance_key in max_instances else 1
    pred_instances = []
    neg_pred_instances = []
    dummy_pred_vector = deepcopy(pred_vector)

    for instance_idx in range(max_num_occurences):
        curr_pred = []
        curr_pred_neg = []
        if len(history_q[-1].children) > 0:
            current_l2_question = history_q[-1].children[-1]
        else:
            history_q[-1].children.append(Question("", "", 2, history_q[-1]))
            current_l2_question = history_q[-1].children[-1]
        if not no_predicted:
            if instance_idx == 0:  # prediction before was positive, otherwise iterate_instances is not called
                q_positive_pred = True
            else:
                # generate follow-up question - if there are more than one occurances
                question = get_question(elem_name, topic_name, area_name, first_instance=False)
                # make prediction
                # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val', args=args)
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                #q_positive_pred = torch.argmax(out[0, [58, 95]]) == 1
                q_positive_pred = True # true so we can move to next level of predictions

            if not q_positive_pred:  # no was predicted for further instance
                if match_instances:
                    curr_pred_neg.extend(NO)
                    dummy_pred_vector.extend(NO)
                    lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(dummy_pred_vector):]) if
                                           key.startswith(
                                               f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                               str(instance_idx))]

                else:
                    pred_vector.extend(NO)
                    report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != 'infos' else f"{area_name}_{topic_name}_yes"
                    report_key = report_keys[len(pred_vector) - 2]
                    if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                            "3") or report_key.endswith("4"):
                        report_key = report_key[:-2]
                    assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                    lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                           key.startswith(
                                               f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                               str(instance_idx))]
                # add -2 to pred_vector for all elements in lower hierarchy ids
                if match_instances:
                    curr_pred_neg.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                    dummy_pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                else:
                    pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                no_predicted = True

            else:

                elem_history.append((question, ["yes"]))
                if match_instances:
                    curr_pred.extend(YES)
                    dummy_pred_vector.extend(YES)
                else:
                    pred_vector.extend(YES)
                    report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != "infos" else f"{area_name}_{topic_name}_yes"
                    report_key = report_keys[len(pred_vector) - 2]
                    if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                            "3") or report_key.endswith("4"):
                        report_key = report_key[:-2]
                    #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"

                for key in instance_keys:  # different "infos" instances
                    if key == "body_region":

                        question = "In which part of the body?"
                        # if len(current_l2_question.children) > 0:
                        #     if not current_l2_question.is_question_present_in_children(question):
                        #         break
                        #     current_l2_question.children.append(Question(key, question, level=3))
                        # else:
                        #     current_l2_question.children.append(Question(key, question, level=3))

                        if not current_l2_question.is_question_present_in_children(question):
                            current_l2_question.children.append(Question(key, question, level=3))


                        # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                        #                                                                          args=args)
                        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                        #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')
                        out = []
                        pred = []
                        #language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        #elem_history.append((question, language_answers))
                    elif key == "localization":
                        question = "In which area?"
                        if not current_l2_question.is_question_present_in_children(question):
                            current_l2_question.children.append(Question(key, question, level=3))
                        # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                        #                                                                          args=args)
                        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                        #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')
                        out = []
                        pred = []
                        #language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        #elem_history.append((question, language_answers))
                    elif key == "attributes":
                        question = "What are the attributes?"
                        if not current_l2_question.is_question_present_in_children(question):
                            current_l2_question.children.append(Question(key, question, level=3))
                        # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                        #                                                                          args=args)
                        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                        #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')
                        out = []
                        pred = []
                        #language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        #elem_history.append((question, language_answers))
                    elif key == "degree":
                        question = "What is the degree?"
                        if not current_l2_question.is_question_present_in_children(question):
                            current_l2_question.children.append(Question(key, question, level=3))
                        # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, elem_history, tokenizer, mode='val',
                        #                                                                          args=args)
                        # # with torch.cuda.amp.autocast():
                        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                        #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                        #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')
                        out = []
                        pred = []
                        #language_answers, pred = get_value(out, infos[key], answer_options)
                        if match_instances:
                            curr_pred.extend(pred)
                            dummy_pred_vector.extend(pred)
                        else:
                            pred_vector.extend(pred)
                            report_key_gen = f"{area_name}_{topic_name}_{elem_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}" if topic_name != "infos" else f"{area_name}_{topic_name}_{key}_{infos[key]['options'][-1]}_{instance_idx}"
                            report_key = report_keys[len(pred_vector) - 1]
                            #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        #elem_history.append((question, language_answers))


        else:
            # just add negative dummy answer until max_instances are filled
            if match_instances:
                curr_pred_neg.extend(NO)
                dummy_pred_vector.extend(NO)
                lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(dummy_pred_vector):]) if
                                       key.startswith(
                                           f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                           str(instance_idx))]

            else:
                pred_vector.extend(NO)
                report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes" if topic_name != "infos" else f"{area_name}_{topic_name}_yes"
                report_key = report_keys[len(pred_vector) - 2]
                if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                        "3") or report_key.endswith("4"):
                    report_key = report_key[:-2]
                #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                       key.startswith(
                                           f"{area_name}_{topic_name}_{elem_name}_" if topic_name != "infos" else f"{area_name}_{topic_name}_") and key.endswith(
                                           str(instance_idx))]
            # add -2 to pred_vector for all elements in lower hierarchy ids
            if match_instances:
                curr_pred_neg.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
                dummy_pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))
            else:
                pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))

        if match_instances:
            if len(curr_pred) > 0:
                pred_instances.append(curr_pred)
            if len(curr_pred_neg) > 0:
                neg_pred_instances.append(curr_pred_neg)
            assert len(curr_pred) > 0 or len(curr_pred_neg) > 0, f"Neither positive nor negative prediction for {area_name}_{topic_name}_{elem_name}"

    if match_instances:
        # find best match between gt_instances and pred_instances (lowest F1 score)
        # don't match predictions to empty gt if there are other gt instances
        num_gt = len(gt_instances)
        num_pred = len(pred_instances)

        if num_gt == 0 or num_pred == 0:
            # add flattened pred_instances, followed by neg_pred_instances to pred_vector in order they were predicted
            pred_vector.extend([item for sublist in pred_instances for item in sublist])
            pred_vector.extend([item for sublist in neg_pred_instances for item in sublist])
            assert pred_vector == dummy_pred_vector

        else:
            # get vectors from target for gt_instances
            instance_len = len(pred_instances[0])
            start_idx = len(pred_vector)  # where the current finding starts
            gt_vectors = []
            for gt_instance in gt_instances:
                gt_vector = report_vector_gt[0, start_idx:start_idx + instance_len]
                start_idx += instance_len
                gt_vectors.append(gt_vector)
                #assert (gt_vector[:2] == torch.tensor(YES)).all()  # should only be positive instances starti

            gt_idxs = list(range(len(gt_vectors)))  # only positive instances
            pred_idxs = list(range(len(pred_instances)))  # only positive instances

            if len(gt_idxs) > len(pred_idxs):
                pass
                #gt_permutations = list(itertools.permutations(gt_idxs, len(pred_idxs)))
                #matchings = [(torch.tensor(gt), torch.tensor(pred_idxs)) for gt in gt_permutations]
            else:
                pass
                #pred_permutations = list(itertools.permutations(pred_idxs, len(gt_idxs)))
                #matchings = [(torch.tensor(gt_idxs), torch.tensor(pred)) for pred in pred_permutations]
                matchings = []
            # Compute F1 scores for all possible matchings
            f1_scores = {}
            for gt_idxs_match, pred_idxs_match in matchings:
                pass
                # get concatenated gt vector in order of gt_idxs
                #gt_vector = torch.cat([gt_vectors[idx] for idx in gt_idxs_match], dim=0)
                # get concatenated pred vector in order of pred_idxs
                #preds_vector = torch.cat([torch.tensor(pred_instances[idx]) for idx in pred_idxs_match], dim=0)
                # compute F1 score
                #f1_scores[(gt_idxs_match, pred_idxs_match)] = f1_score(gt_vector, preds_vector, average='macro')

            # get best matching
            best_matching = max(f1_scores, key=f1_scores.get)
            gt_idxs_best, pred_idxs_best = best_matching

            # construct what to add to pred_vector using best_matching
            # sort gt_idxs and pred_idxs such that gt_idxs is in ascending order
            gt_idxs_best, pred_idxs_best = zip(*sorted(zip(gt_idxs_best, pred_idxs_best)))
            unmatched_pred_idxs = [idx for idx in range(len(pred_instances)) if idx not in pred_idxs_best]
            # create flattened pred vector
            # in ascending order of gt_idxs, if idx was matched add matched pred to vector, else add one of the neg_pred_instances
            for idx in range(max_num_occurences):
                if idx in gt_idxs_best:
                    pred_idx = pred_idxs_best[gt_idxs_best.index(idx)]
                    pred_vector.extend(pred_instances[pred_idx])
                else:
                    if len(unmatched_pred_idxs) > 0:
                        pred_idx = unmatched_pred_idxs.pop(0)
                        pred_vector.extend(pred_instances[pred_idx])
                    else:
                        # add one of the neg_pred_instances
                        pred_vector.extend(neg_pred_instances[0])
                        neg_pred_instances = neg_pred_instances[1:]

    return pred_vector


def iterate_area_VQA(area, area_name, max_instances, pred_vector, report_keys, answer_options, report_vector_gt,
                     match_instances):

    area_questions = []

    for topic_name, topic in area.items():
        if topic_name == 'area':
            continue

        # get prediction for topic question
        area_question = get_topic_question(topic_name, area_name)

        ### AD
        area_questions.append({area_question: None})
        # with torch.cuda.amp.autocast():


        ### AD - always positive prediction = go through all questions
        area_positive_pred = 1 == 1  # yes was predicted

        if not area_positive_pred:  # we predicted no -> don't answer any following questions
            #pred_vector.extend(NO)

            ### AD
            area_questions[len(area_questions) - 1] = {area_question: "No"}

            # get report_key for current question
            report_key_gen = f"{area_name}_{topic_name}_yes"
            report_key = report_keys[len(pred_vector) - 2]
            if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith("3") or report_key.endswith(
                    "4"):
                report_key = report_key[:-2]
            assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
            lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                   key.startswith(f"{area_name}_{topic_name}_")]

            # add -2 to pred_vector for all elements in lower hierarchy ids
            pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))

        else:  # predicted yes -> iterate through all elements
            history = []

            ### AD
            area_questions[len(area_questions) - 1] = {area_question: "Yes"}
            if topic_name != 'infos':
                pred_vector.extend(YES)
                report_key_gen = f"{area_name}_{topic_name}_yes"
                report_key = report_keys[len(pred_vector) - 2]
                if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                        "3") or report_key.endswith("4"):
                    report_key = report_key[:-2]
                #assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                history.append((area_question, ["yes"]))
                #history_ad[f"{area_name}_{topic_name}"] = [{"L1": area_question}]
                l1_question = Question(f"{area_name}_{topic_name}", area_question, level=1)
                history_ad[f"{area_name}_{topic_name}"] = {"L1": area_question}
                history_q.append(l1_question)



                for elem_name, elem in area[topic_name].items():
                    ###
                    ### If first time a L2 question is asked
                    if len(history_ad[f"{area_name}_{topic_name}"])<2:
                        history_ad[f"{area_name}_{topic_name}"]["L2"] = [[elem_name], []]

                    question = get_question(elem_name, topic_name, area_name, first_instance=True)
                    history_ad[f"{area_name}_{topic_name}"]["L2"][1].append(question)
                    history_q[-1].children.append(Question(elem_name, question, level=2))
                    # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, tokenizer, mode='val', args=args)
                    # # with torch.cuda.amp.autocast():
                    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # out, _ = model(img=img.to(device), input_ids=torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                    #                q_attn_mask=torch.tensor(q_attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                    #                attn_mask=torch.tensor(attn_mask, dtype=torch.long, device=device).unsqueeze(0),
                    #                token_type_ids_q=token_type_ids.unsqueeze(0), mode='val')

                    elem_positive_pred = 1 == 1

                    if not elem_positive_pred:
                        pred_vector.extend(NO)
                        lower_hierarchy_ids = [(idx, key) for idx, key in enumerate(report_keys[len(pred_vector):]) if
                                               key.startswith(f"{area_name}_{topic_name}_{elem_name}_")]
                        report_key_gen = f"{area_name}_{topic_name}_{elem_name}_yes"
                        report_key = report_keys[len(pred_vector) - 2]
                        if report_key.endswith("0") or report_key.endswith("1") or report_key.endswith("2") or report_key.endswith(
                                "3") or report_key.endswith("4"):
                            report_key = report_key[:-2]
                        assert report_key == report_key_gen, f"Report key {report_key} does not match generated report key {report_key_gen}"
                        # add -2 to pred_vector for all elements in lower hierarchy ids
                        pred_vector.extend([NOT_PREDICTED] * len(lower_hierarchy_ids))


                    else:  # positive prediction
                        pred_vector = iterate_instances_VQA(elem, question, elem_name, topic_name, area_name, history,
                                                             pred_vector, report_keys, max_instances, answer_options, report_vector_gt,
                                                             match_instances)
                        print("hi")
            else:
                ### Look into that - infos only ?
                question = get_question(area_name, topic_name, area_name, first_instance=True)

                history_ad[f"{area_name}_{topic_name}"] = [{"L1": question}]
                l1_question = Question(f"{area_name}_{topic_name}", area_question, level=1)
                history_q.append(l1_question)
                ### Ad
                ### L3 question
                pred_vector = iterate_instances_VQA(topic, question, area_name, topic_name, area_name, history,
                                                     pred_vector, report_keys, max_instances, answer_options, report_vector_gt, match_instances)


    # with open(f"{report_id}.txt", "w", encoding="utf-8") as f:
    #     for question in area_questions:
    #         f.write(f"{question}\n")

    return pred_vector




# with open('report_1_L1_L2_questions.txt', 'w') as f:
#     for key in history_ad:
#         f.write(f"{key}\n")
#
#         for l1 in history_ad[key]:
#
#             if l1=="L1":
#                 f.write(f"\t{history_ad[key][l1]}\n")
#             if l1=="L2":
#                 for question in history_ad[key][l1]:
#
#                     f.write(f"\t\t{question}\n")

# with open('report_1_L1_L2_L3questions.txt', 'w') as f:
#     for question in history_q:
#         # only 1 question in L1
#         f.write(f"{question.questions[0]}\t\t\t\t{question.key}\n")
#
#         # L2 questions
#         for l2_children in question.children:
#             #f.write(f"\t{l2_children.key}\n")
#             for l2_question in l2_children.questions:
#                 f.write(f"\t{l2_question}\t\t\t\t{question.key}_{l2_children.key}\n")
#                 for l3_child in l2_children.children:
#                     #f.write(f"\t\t{l3_child.key}\n")
#                     for l3_question in l3_child.questions:
#                         f.write(f"\t\t{l3_question} \t\t\t{question.key}_{l2_children.key}_{l3_child.key}\n")


# Generate a reference list of hierarchical questions and their keys
def generate_questions_list(file_name)->list:
    # A list of l1 questions
    l1_questions = []
    with open(file_name, 'r') as file:
        report = csv.reader(file)
        for row in report:

            # If the row contains a level 1 question

            last_question = None
            history_questions = None
            parent = None
            # if row[1] == '1':
            #     questions.append(Question(row[2], row[0].strip(), row[1]))
            # # If the row contains a level 2 question, find last
            # # level 1 question and nest it under it
            # if row[1] == '2':
            #     questions[-1].children.append(Question(row[2], row[0].strip(), row[1]))
            # # If the row contains a level 3 question, find last
            # # level 2 question - it's going to be the child of the
            # # last l1 question and put the l3 question under it
            # if row[1] == '3':
            #     questions[-1].children[-1].children.append(Question(row[2], row[0].strip(), row[1]))

            # row 0 - Question - "Are there any foreign objects?"
            # row 1 - Level (as str) - '1'
            # row 2 - Key (as str) - 'foreign objects_objects'
            if row[1] == '1':
                last_question = l1_questions
                #history_questions = [l1_questions[-1].questions[-1]]
            # If the row contains a level 2 question, find last
            # level 1 question and nest it under it
            if row[1] == '2':
                last_question = l1_questions[-1].children
                history_questions = [l1_questions[-1].questions[-1]]
                parent = l1_questions[-1]
            # If the row contains a level 3 question, find last
            # level 2 question - it's going to be the child of the
            # last l1 question and put the l3 question under it
            if row[1] == '3':
                last_question = l1_questions[-1].children[-1].children
                parent = l1_questions[-1].children[-1]

            last_question.append(Question(row[2], row[0].strip(), row[1], parent))

    return l1_questions




#positive_l1_reports = []
def get_reports_with_positive_l1_questions_QPAIRS():
    positive_l1_reports = []

    for data_dir in DATA_DIRS_QAPAIRS:
        sorted_report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))

        for report_file in sorted_report_files:
            with open(data_dir / report_file, 'r') as json_file:
                report_as_list = json.load(json_file)

                for report_question in report_as_list:
                    # if the answer to the L1 question is yes and it is a L1 question (doesn't have any history)
                    # question is a list of size 4
                    # [0] str: is the question
                    # [1] list: is the answer to the question ['yes'] / ['upper-lobe', 'middle-lobe']
                    # [2] list: is the history = questions leading up to this one. Only populated
                    # for questions of level 2 + level 3
                    # [3] dict: of metadata
                    # [3]['answer_type'] str: 'single_choice' / 'multiple_choice' ?
                    # [3]['options'] list: ['yes', 'no', 'upper-left' ...]
                    # [3]['path'] str: 'foreign objects_objects'
                    if report_question[1][0] == 'yes' and len(report_question[2]) == 0:

                        # if multiple L1 questions are answered yes in the same report
                        if len(positive_l1_reports) > 0 and positive_l1_reports[-1]["report_file"] == report_file:
                            positive_l1_reports[-1]["question"].append(report_question[0])
                            positive_l1_reports[-1]["answer"].append(report_question[1][0])
                            positive_l1_reports[-1]["key"].append(report_question[3]['path'])
                        else:
                            positive_l1_reports.append({"report_file": report_file,
                                                    "question": [report_question[0]],
                                                    "answer": [report_question[1][0]],
                                                    "key": [report_question[3]['path']]})
            print(f"file {report_file} has been processed")

    return positive_l1_reports




def get_reports_with_positive_l1_question_QAPAIRS(l1_question:str):
    positive_l1_reports = []

    for data_dir in DATA_DIRS_QAPAIRS:
        sorted_report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))

        for report_file in sorted_report_files:
            with open(data_dir / report_file, 'r') as json_file:
                report_as_list = json.load(json_file)

                for question in report_as_list:
                    # if the answer to the L1 question is yes and it is a L1 question (doesn't have any history)
                    # question is a list of size 4
                    # [0] str: is the question
                    # [1] list: is the answer to the question ['yes'] / ['upper-lobe', 'middle-lobe']
                    # [2] list: is the history = questions leading up to this one. Only populated
                    # for questions of level 2 + level 3
                    # [3] dict: of metadata
                    # [3]['answer_type'] str: 'single_choice' / 'multiple_choice' ?
                    # [3]['options'] list: ['yes', 'no', 'upper-left' ...]
                    # [3]['path'] str: 'foreign objects_objects'
                    #### READ - the above commented version was changed to match from an actual question
                    # like - 'Is there anything abnormal in the abdoment?' to match the path of that question
                    # the l1 question passed to this function was also changed to reflect that
                    if question[1][0] == 'yes' and question[0] == l1_question:
                        positive_l1_reports.append(report_file)
                        break
                    # if question[1][0] == 'yes' and question[3]['path'] == l1_question:
                    #     positive_l1_reports.append(report_file)
                    #     break
            print(f"file {report_file} has been processed")

    return positive_l1_reports



class Mode(Enum):
    VECTORIZED_ANSWERS = 1
    QA_PAIRS = 2

#positive_l1_reports = []
def get_reports_with_positive_l1_questions(mode=Mode.VECTORIZED_ANSWERS):
    positive_l1_reports = []

    if mode == Mode.QA_PAIRS:
        for data_dir in DATA_DIRS_QAPAIRS:
            sorted_report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))

            for report_file in sorted_report_files:
                with open(data_dir / report_file, 'r') as json_file:
                    report_as_list = json.load(json_file)

                    for report_question in report_as_list:
                        # if the answer to the L1 question is yes and it is a L1 question (doesn't have any history)
                        # question is a list of size 4
                        # [0] str: is the question
                        # [1] list: is the answer to the question ['yes'] / ['upper-lobe', 'middle-lobe']
                        # [2] list: is the history = questions leading up to this one. Only populated
                        # for questions of level 2 + level 3
                        # [3] dict: of metadata
                        # [3]['answer_type'] str: 'single_choice' / 'multiple_choice' ?
                        # [3]['options'] list: ['yes', 'no', 'upper-left' ...]
                        # [3]['path'] str: 'foreign objects_objects'
                        if report_question[1][0] == 'yes' and len(report_question[2]) == 0:

                            # if multiple L1 questions are answered yes in the same report
                            # if len(positive_l1_reports) > 0 and positive_l1_reports[-1]["report_file"] == report_file:
                            #     positive_l1_reports[-1]["question"].append(question[0])
                            #     positive_l1_reports[-1]["answer"].append(question[1][0])
                            #     positive_l1_reports[-1]["key"].append(question[3]['path'])
                            # else:
                            #     positive_l1_reports.append({"report_file": report_file,
                            #                             "question": [question[0]],
                            #                             "answer": [question[1][0]],
                            #                             "key": [question[3]['path']]})
                                positive_l1_reports.append(report_file)
                                break
                print(f"file {report_file} has been processed")

    elif mode == Mode.VECTORIZED_ANSWERS:
        for data_dir in DATA_DIRS_VECTORIZED_ANSWERS:
            sorted_report_files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('.')[0]))
            for report_file in sorted_report_files:

                with open(data_dir / report_file, 'r') as json_file:
                    report_as_list = json.load(json_file)

                    for report_question in report_as_list:
                        if report_question.endswith('yes'):
                            if report_as_list[report_question] == True:
                                #positive_l1_reports.append(report_file)
                                break
                        if "infos" in report_question:
                            if report_question.endswith('yes_0'):
                                if report_as_list[report_question] == True:
                                    # positive_l1_reports.append(report_file)
                                    break
                    else:
                        continue
                        #positive_l1_reports.append(report_file)

                    positive_l1_reports.append(report_file)
                    print(f"file {report_file} has been processed")


    return positive_l1_reports


# given a list of reports that have an answer yes for ANY L1 question,
# get those reports that are positive for a given category
def get_positive_l1_reports_for_category(positive_l1_reports:list, category:str)->list:
    matches = []
    for report in positive_l1_reports:
        if category in report["key"]:
            # the name of the report file - 2.json
            matches.append({"report_file": report["report_file"],
                            "question": report["question"][report["key"].index(category)]}) # the question in human form


    return matches




# Given a list lf L1 Questions, return the L3 Question keys and their corresponding
# questions
def get_l3_questions(questions: list)->dict:
    l3_questions = {}
    for l1_question in questions:
        for l2_question in l1_question.children:
            for l3_question in l2_question.children:
                l3_questions[l3_question.key] = [l3_question.questions[0], l2_question.questions[0], l1_question.questions[0]] # there will only ever be 1 L3 question = 0th in the list


                # question = l3_question
                # while question.parent is not None:




    return l3_questions

def get_l1l2l3_questions(questions: list)->dict:
    l1l2l3_questions = {}
    for l1_question in questions:
        l1l2l3_questions[l1_question.key] = [l1_question.questions[0]]
        for l2_question in l1_question.children:
            l1l2l3_questions[l2_question.key] = [l2_question.questions[0], l1_question.questions[0]]
            for l3_question in l2_question.children:
                l1l2l3_questions[l3_question.key] = [l3_question.questions[0], l2_question.questions[0], l1_question.questions[0]] # there will only ever be 1 L3 question = 0th in the list


                # question = l3_question
                # while question.parent is not None:




    return l1l2l3_questions



if __name__ == "__main__":

    with open(Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\new_reports\\2.json"), 'r') as f:
        report = json.loads(f.read())
    with open(Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\train_vectorized_answers\\2.json"), 'r') as f:
        report_keys = list(json.load(f).keys())  # same for all reports
    with open(Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\answer_options.json"), 'r') as f:
        answer_options = list(json.load(f).keys())
    with open(Path("E:\\Development\\ad_Rad-ReStruct\\data\\radrestruct\\max_instances.json"), 'r') as f:
        max_instances = json.loads(f.read())

    for area in report:
        if "sub_areas" in area:
            for sub_area_name, sub_area in area["sub_areas"].items():
                pred_vector = iterate_area_VQA(sub_area, sub_area_name,
                                               max_instances, pred_vector, report_keys,
                                               answer_options, report_vector_gt, match_instances)

        else:
            pred_vector = iterate_area_VQA(area, area['area'], max_instances,
                                           pred_vector, report_keys,
                                           answer_options, report_vector_gt, match_instances)
    #assert len(pred_vector) == len(report_keys)
    preds.append(pred_vector)



    #### GENERATE THE QUESTIONS FILE
    with open('report_questions_reference_infos_trainvaltest.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for question in history_q:
            writer.writerow([question.questions[0], 1, question.key])

            for l2_children in question.children:
                for l2_question in l2_children.questions:

                    # If statement for infos such as lymph_nodes_infos = only L1 + L2 present
                    if l2_children.key == '':
                        writer.writerow(["      " + l2_question, 2, f"{question.key}"])
                    else:
                        writer.writerow(["      " + l2_question, 2, f"{question.key}_{l2_children.key}"])

                    for l3_child in l2_children.children:
                        for l3_question in l3_child.questions:
                            # If statement for infos such as lymph_nodes_infos = only L1 + L2 present
                            if l2_children.key == '':
                                writer.writerow(["            " + l3_question, 3,
                                     f"{question.key}_{l3_child.key}"])
                            else:
                                writer.writerow(["            " + l3_question, 3,
                                     f"{question.key}_{l2_children.key}_{l3_child.key}"])

    print("questions generated!")

    #reports = get_reports_with_positive_l1_question("Is there anything abnormal in the abdomen?")
    reports = get_reports_with_positive_l1_question_QAPAIRS("Are there any signs in the abdomen?")
    reports = get_reports_with_positive_l1_questions_QPAIRS()
    l1_reports_qa = get_reports_with_positive_l1_questions(Mode.QA_PAIRS)
    l1_reports_vec = get_reports_with_positive_l1_questions(Mode.VECTORIZED_ANSWERS)

    print(len(l1_reports_qa))
    print(len(l1_reports_vec))

    question = generate_questions_list("report_questions_reference_infos_trainvaltest.csv")
    print(f"Positive L1 reports: {len(l1_reports_qa)}")
    #print(len(generate_questions_list(REPORT_QUESTIONS_REFERENCE)))
    l3_questions = get_l3_questions(question)
    print(len(get_positive_l1_reports_for_category(reports, 'cardiovascular system_signs')))

