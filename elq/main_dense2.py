# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys
from elq.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer, DenseIVFFlatIndexer

import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored
import torch.nn.functional as F

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from elq.biencoder.biencoder import BiEncoderRanker, load_biencoder, to_bert_input
from elq.biencoder.data_process import (
    process_mention_data,
    get_context_representation_single_mention,
    get_candidate_representation,
)
import elq.candidate_ranking.utils as utils
import math

from elq.vcg_utils.measures import entity_linking_tp_with_overlap
from elq.biencoder.utils import batch_reshape_mask_left

import os
import sys
from tqdm import tqdm
import pdb
import time
import sentencepiece as spm


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from sentenpiece_test import bert_to_ads

def _print_colorful_text(input_tokens, tokenizer, pred_triples):
    """
    pred_triples:
        Assumes no overlapping triples
    """
    sort_idxs = sorted(range(len(pred_triples)), key=lambda idx: pred_triples[idx][1])

    init()  # colorful output
    msg = ""
    if pred_triples and (len(pred_triples) > 0):
        msg += tokenizer.decode(input_tokens[0 : int(pred_triples[sort_idxs[0]][1])])
        for i, idx in enumerate(sort_idxs):
            triple = pred_triples[idx]
            msg += " " + colored(
                tokenizer.decode(input_tokens[int(triple[1]) : int(triple[2])]),
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if i < len(sort_idxs) - 1:
                msg += " " + tokenizer.decode(input_tokens[
                    int(triple[2]) : int(pred_triples[sort_idxs[i + 1]][1])
                ])
            else:
                msg += " " + tokenizer.decode(input_tokens[int(triple[2]) : ])
    else:
        msg = tokenizer.decode(input_tokens)
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(all_entity_preds, pred_triples, id2text, id2wikidata):
    sort_idxs = sorted(range(len(pred_triples)), key=lambda idx: pred_triples[idx][1])
    for idx in sort_idxs:
        print(colored(all_entity_preds[0]['pred_tuples_string'][idx][1], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
        if pred_triples[idx][0] in id2wikidata:
            print("    Wikidata ID: {}".format(id2wikidata[pred_triples[idx][0]]))
        print("    Title: {}".format(all_entity_preds[0]['pred_tuples_string'][idx][0]))
        print("    Score: {}".format(str(all_entity_preds[0]['scores'][idx])))
        print("    Triple: {}".format(str(pred_triples[idx])))
        print("    Text: {}".format(id2text[pred_triples[idx][0]]))


def _load_candidates(
    entity_catalogue, entity_encoding,
    faiss_index="none", index_path=None,
    logger=None,
):
    if faiss_index == "none":
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        elif faiss_index == "ivfflat":
            indexer = DenseIVFFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw,ivfflat.")
        indexer.deserialize_from(index_path)

    candidate_encoding = torch.load(entity_encoding)

    if not os.path.exists("/data/vmorisetty/models/elq_models/id2title.json"):
        id2title = {}
        id2text = {}
        id2wikidata = {}
        local_idx = 0
        with open(entity_catalogue, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                entity = json.loads(line)
                id2title[str(local_idx)] = entity["title"]
                id2text[str(local_idx)] = entity["text"]
                if "kb_idx" in entity:
                    id2wikidata[str(local_idx)] = entity["kb_idx"]
                local_idx += 1
        json.dump(id2title, open("/data/vmorisetty/models/elq_models/id2title.json", "w"))
        json.dump(id2text, open("/data/vmorisetty/models/elq_models/id2text.json", "w"))
        json.dump(id2wikidata, open("/data/vmorisetty/models/elq_models/id2wikidata.json", "w"))
    else:
        if logger: logger.info("Loading id2title")
        id2title = json.load(open("/data/vmorisetty/models/elq_models/id2title.json"))
        if logger: logger.info("Loading id2text")
        id2text = json.load(open("/data/vmorisetty/models/elq_models/id2text.json"))
        if logger: logger.info("Loading id2wikidata")
        id2wikidata = json.load(open("/data/vmorisetty/models/elq_models/id2wikidata.json"))

    return (
        candidate_encoding, indexer, 
        id2title, id2text, id2wikidata,
    )


def _get_test_samples(
    test_filename, test_entities_path, logger,
):
    """
    Parses jsonl format with one example per line
    Each line of the following form

    IF HAVE LABELS
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
        "mentions": [[19, 23], [7, 15]],
        "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
        "tokenized_mention_idxs": [[4, 5], [2, 3]],
        "label_id": [10902, 28422],
        "wikidata_id": ["Q1397", "Q132050"],
        "entity": ["Ohio", "Governor"],
        "label": [list of wikipedia descriptions]
    }

    IF NO LABELS (JUST PREDICTION)
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    if logger: logger.info("Loading test samples")
    test_samples = []
    unknown_entity_samples = []
    num_unknown_entity_samples = 0
    num_no_gold_entity = 0
    ner_errors = 0

    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        sample_idx = 0
        do_setup_samples = True
        for i, line in enumerate(lines):
            record = json.loads(line)
            test_samples.append(record)

    return test_samples, num_unknown_entity_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params, logger):
    """
    Samples: list of examples, each of the form--

    IF HAVE LABELS
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
        "mentions": [[19, 23], [7, 15]],
        "tokenized_text_ids": [2040, 2003, 3099, 1997, 4058, 2249, 1029],
        "tokenized_mention_idxs": [[4, 5], [2, 3]],
        "label_id": [10902, 28422],
        "wikidata_id": ["Q1397", "Q132050"],
        "entity": ["Ohio", "Governor"],
        "label": [list of wikipedia descriptions]
    }

    IF NO LABELS (JUST PREDICTION)
    {
        "id": "WebQTest-12",
        "text": "who is governor of ohio 2011?",
    }
    """
    if 'label_id' in samples[0]:
        # have labels
        tokens_data, tensor_data_tuple, _ = process_mention_data(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=biencoder_params["max_context_length"],
            max_cand_length=biencoder_params["max_cand_length"],
            silent=False,
            logger=logger,
            debug=biencoder_params["debug"],
            add_mention_bounds=(not biencoder_params.get("no_mention_bounds", False)),
            params=biencoder_params,
        )
    else:
        samples_text_tuple = []
        max_seq_len = 0
        for sample in samples:
            samples_text_tuple
            # truncate the end if the sequence is too long...
            encoded_sample = [101] + tokenizer.encode(sample['text'])[:biencoder_params["max_context_length"]-2] + [102]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            samples_text_tuple.append(encoded_sample + [0 for _ in range(biencoder_params["max_context_length"] - len(encoded_sample))])

            # print(samples_text_tuple)

        tensor_data_tuple = [torch.tensor(samples_text_tuple)]
    tensor_data = TensorDataset(*tensor_data_tuple)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(
    args, biencoder, dataloader, candidate_encoding, samples,
    num_cand_mentions=50, num_cand_entities=10,
    device="cpu", sample_to_all_context_inputs=None,
    threshold=0.0, indexer=None,
):
    """
    Returns: tuple
        labels (List[int]) [(max_num_mentions_gold) x exs]: gold labels -- returns None if no labels
        nns (List[Array[int]]) [(# of pred mentions, cands_per_mention) x exs]: predicted entity IDs in each example
        dists (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: scores of each entity in nns
        pred_mention_bounds (List[Array[int]]) [(# of pred mentions, 2) x exs]: predicted mention boundaries in each examples
        mention_scores (List[Array[float]]) [(# of pred mentions,) x exs]: mention score logit
        cand_scores (List[Array[float]]) [(# of pred mentions, cands_per_mention) x exs]: candidate score logit
    """
    biencoder.model.eval()
    biencoder_model = biencoder.model
    if hasattr(biencoder.model, "module"):
        biencoder_model = biencoder.model.module

    context_inputs = []
    nns = []
    dists = []
    mention_dists = []
    pred_mention_bounds = []
    mention_scores = []
    cand_scores = []
    sample_idx = 0
    ctxt_idx = 0
    label_ids = None
    for step, batch in enumerate(tqdm(dataloader)):
        context_input = batch[0].to(device)
        mask_ctxt = context_input != biencoder.NULL_IDX
        with torch.no_grad():
            start = time.time()
            context_outs = biencoder.encode_context(
                context_input, num_cand_mentions=num_cand_mentions, topK_threshold=threshold,
            )
            end = time.time()
            print("context encode time: " + str(end-start))
            embedding_ctxt = context_outs['mention_reps']
            left_align_mask = context_outs['mention_masks']>0
            chosen_mention_logits = context_outs['mention_logits']
            chosen_mention_bounds = context_outs['mention_bounds']

            filtered_mention_bounds = []
            filtered_mention_scores = []
            filtered_embeddings_ctxt = []
            filtered_mask = []
            filtered_tokens = []
            entity_text = []
            filtered_token_ids=[]
            mention_threshold=-0.6931
            
            for idx in range(len(batch[0])):
                highest_logit = -999
                highest_index = -1
                for i, logit in enumerate(chosen_mention_logits[idx]):
                    if(logit > highest_logit):
                        highest_index = i
                        highest_logit = logit
                

                tokens = []
                token_ids = []
                #output all tokens
                for i in range(len(context_input[idx])):
                    if(context_input[idx][i] == 101):
                        continue
                    elif(context_input[idx][i] == 102):
                        break
                    else:
                        tokens.append(tokenizer.decode(context_input[idx][i:i+1]))
                        token_ids.append(context_input[idx][i:i+1])

                filtered_tokens.append(tokens) 
                filtered_token_ids.append(token_ids)

                if(highest_logit > mention_threshold):
                    filtered_mask.append(True)
                    adjusted_mention_bounds = [int(chosen_mention_bounds[idx][highest_index][0].data.cpu().numpy() -2),
                                            int(chosen_mention_bounds[idx][highest_index][1].data.cpu().numpy() -2)]
                    filtered_mention_bounds.append(adjusted_mention_bounds)                        
                    #vec1024d = np.reshape(embedding_ctxt[idx][highest_index].data.cpu().numpy(),(-1, 1024))
                    #vec64d = np.reshape(PCAMat.apply(vec1024d), (64,))
                    filtered_embeddings_ctxt.append(embedding_ctxt[idx][highest_index].data.cpu().numpy())
                    filtered_mention_scores.append(highest_logit.data.cpu().numpy())
                    
                    # #output only span tokens
                    # for i in range(chosen_mention_bounds[idx][highest_index][0], chosen_mention_bounds[idx][highest_index][1]+1):
                    #     tokens.append(tokenizer.decode(context_input[idx][i:i+1]))
                    entity_text.append(tokenizer.decode(context_input[idx][chosen_mention_bounds[idx][highest_index][0]:chosen_mention_bounds[idx][highest_index][1]+1]))
                else:
                    filtered_mask.append(False)
                    empty_tensor1 = torch.empty(0,2)
                    filtered_mention_bounds.append(empty_tensor1.data.cpu().numpy().tolist())
                    empty_tensor2 = torch.empty(0,1024)
                    filtered_embeddings_ctxt.append(empty_tensor2.data.cpu().numpy())
                    filtered_mention_scores.append(highest_logit)
                    entity_text.append("")
                   


    return filtered_embeddings_ctxt, filtered_mention_bounds, filtered_mention_scores, filtered_mask, filtered_tokens,entity_text

            


def get_predictions(
    args, dataloader, biencoder_params, samples, nns, dists, mention_scores, cand_scores,
    pred_mention_bounds, id2title, threshold=-2.9, mention_threshold=-0.6931,
):
    """
    Arguments:
        args, dataloader, biencoder_params, samples, nns, dists, pred_mention_bounds
    Returns:
        all_entity_preds,
        num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    """

    # save biencoder predictions and print precision/recalls
    num_correct_weak = 0
    num_correct_strong = 0
    num_predicted = 0
    num_gold = 0
    num_correct_weak_from_input_window = 0
    num_correct_strong_from_input_window = 0
    num_gold_from_input_window = 0
    all_entity_preds = []

    f = errors_f = None
    if getattr(args, 'save_preds_dir', None) is not None:
        save_biencoder_file = os.path.join(args.save_preds_dir, 'biencoder_outs.jsonl')
        f = open(save_biencoder_file, 'w')
        errors_f = open(os.path.join(args.save_preds_dir, 'biencoder_errors.jsonl'), 'w')

    # nns (List[Array[int]]) [(num_pred_mentions, cands_per_mention) x exs])
    # dists (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # pred_mention_bounds (List[Array[int]]) [(num_pred_mentions, 2) x exs]
    # cand_scores (List[Array[float]]) [(num_pred_mentions, cands_per_mention) x exs])
    # mention_scores (List[Array[float]]) [(num_pred_mentions,) x exs])
    for batch_num, batch_data in enumerate(dataloader):
        batch_context = batch_data[0]
        if len(batch_data) > 1:
            _, batch_cands, batch_label_ids, batch_mention_idxs, batch_mention_idx_masks = batch_data
        for b in range(len(batch_context)):
            i = batch_num * biencoder_params['eval_batch_size'] + b
            sample = samples[i]
            input_context = batch_context[b][batch_context[b] != 0].tolist()  # filter out padding

            # (num_pred_mentions, cands_per_mention)
            scores = dists[i] if args.threshold_type == "joint" else cand_scores[i]
            cands_mask = (scores[:,0] == scores[:,0])
            pred_entity_list = nns[i][cands_mask]
            if len(pred_entity_list) > 0:
                e_id = pred_entity_list[0]
            distances = scores[cands_mask]
            # (num_pred_mentions, 2)
            entity_mention_bounds_idx = pred_mention_bounds[i][cands_mask]
            utterance = sample['text']

            if args.threshold_type == "joint":
                # THRESHOLDING
                assert utterance is not None
                top_mentions_mask = (distances[:,0] > threshold)
            elif args.threshold_type == "top_entity_by_mention":
                top_mentions_mask = (mention_scores[i] > mention_threshold)
            elif args.threshold_type == "thresholded_entity_by_mention":
                top_mentions_mask = (distances[:,0] > threshold) & (mention_scores[i] > mention_threshold)
    
            _, sort_idxs = torch.tensor(distances[:,0][top_mentions_mask]).sort(descending=True)
            # cands already sorted by score
            all_pred_entities = pred_entity_list[:,0][top_mentions_mask]
            e_mention_bounds = entity_mention_bounds_idx[top_mentions_mask]
            chosen_distances = distances[:,0][top_mentions_mask]
            if len(all_pred_entities) >= 2:
                all_pred_entities = all_pred_entities[sort_idxs]
                e_mention_bounds = e_mention_bounds[sort_idxs]
                chosen_distances = chosen_distances[sort_idxs]

            # prune mention overlaps
            e_mention_bounds_pruned = []
            all_pred_entities_pruned = []
            chosen_distances_pruned = []
            mention_masked_utterance = np.zeros(len(input_context))
            # ensure well-formed-ness, prune overlaps
            # greedily pick highest scoring, then prune all overlapping
            for idx, mb in enumerate(e_mention_bounds):
                mb[1] += 1  # prediction was inclusive, now make exclusive
                # check if in existing mentions
                if args.threshold_type != "top_entity_by_mention" and mention_masked_utterance[mb[0]:mb[1]].sum() >= 1:
                    continue
                e_mention_bounds_pruned.append(mb)
                all_pred_entities_pruned.append(all_pred_entities[idx])
                chosen_distances_pruned.append(float(chosen_distances[idx]))
                mention_masked_utterance[mb[0]:mb[1]] = 1

            input_context = input_context[1:-1]  # remove BOS and sep
            pred_triples = [(
                str(all_pred_entities_pruned[j]),
                int(e_mention_bounds_pruned[j][0]) - 1,  # -1 for BOS
                int(e_mention_bounds_pruned[j][1]) - 1,
            ) for j in range(len(all_pred_entities_pruned))]

            entity_results = {
                "id": sample["id"],
                "text": sample["text"],
                "scores": chosen_distances_pruned,
            }

            if 'label_id' in sample:
                # Get LABELS
                input_mention_idxs = batch_mention_idxs[b][batch_mention_idx_masks[b]].tolist()
                input_label_ids = batch_label_ids[b][batch_label_ids[b] != -1].tolist()
                assert len(input_label_ids) == len(input_mention_idxs)
                gold_mention_bounds = [
                    sample['text'][ment[0]-10:ment[0]] + "[" + sample['text'][ment[0]:ment[1]] + "]" + sample['text'][ment[1]:ment[1]+10]
                    for ment in sample['mentions']
                ]

                # GET ALIGNED MENTION_IDXS (input is slightly different to model) between ours and gold labels -- also have to account for BOS
                gold_input = sample['tokenized_text_ids']
                # return first instance of my_input in gold_input
                for my_input_start in range(len(gold_input)):
                    if (
                        gold_input[my_input_start] == input_context[0] and
                        gold_input[my_input_start:my_input_start+len(input_context)] == input_context
                    ):
                        break

                # add alignment factor (my_input_start) to predicted mention triples
                pred_triples = [(
                    triple[0],
                    triple[1] + my_input_start, triple[2] + my_input_start,
                ) for triple in pred_triples]
                gold_triples = [(
                    str(sample['label_id'][j]),
                    sample['tokenized_mention_idxs'][j][0], sample['tokenized_mention_idxs'][j][1],
                ) for j in range(len(sample['label_id']))]
                num_overlap_weak, num_overlap_strong = entity_linking_tp_with_overlap(gold_triples, pred_triples)
                num_correct_weak += num_overlap_weak
                num_correct_strong += num_overlap_strong
                num_predicted += len(all_pred_entities_pruned)
                num_gold += len(sample["label_id"])

                # compute number correct given the input window
                pred_input_window_triples = [(
                    str(all_pred_entities_pruned[j]),
                    int(e_mention_bounds_pruned[j][0]), int(e_mention_bounds_pruned[j][1]),
                ) for j in range(len(all_pred_entities_pruned))]
                gold_input_window_triples = [(
                    str(input_label_ids[j]),
                    input_mention_idxs[j][0], input_mention_idxs[j][1] + 1,
                ) for j in range(len(input_label_ids))]
                num_overlap_weak_window, num_overlap_strong_window = entity_linking_tp_with_overlap(gold_input_window_triples, pred_input_window_triples)
                num_correct_weak_from_input_window += num_overlap_weak_window
                num_correct_strong_from_input_window += num_overlap_strong_window
                num_gold_from_input_window += len(input_mention_idxs)

                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "gold_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(sample['tokenized_text_ids'][triple[1]:triple[2]])]
                        for triple in gold_triples
                    ],
                    "pred_triples": pred_triples,
                    "gold_triples": gold_triples,
                    "tokens": input_context,
                })

                if errors_f is not None and (num_overlap_weak != len(gold_triples) or num_overlap_weak != len(pred_triples)):
                    errors_f.write(json.dumps(entity_results) + "\n")
            else:
                entity_results.update({
                    "pred_tuples_string": [
                        [id2title[triple[0]], tokenizer.decode(input_context[triple[1]:triple[2]])]
                        for triple in pred_triples
                    ],
                    "pred_triples": pred_triples,
                    "tokens": input_context,
                })

            all_entity_preds.append(entity_results)
            if f is not None:
                f.write(
                    json.dumps(entity_results) + "\n"
                )
    
    if f is not None:
        f.close()
        errors_f.close()
    return (
        all_entity_preds, num_correct_weak, num_correct_strong, num_predicted, num_gold,
        num_correct_weak_from_input_window, num_correct_strong_from_input_window, num_gold_from_input_window
    )


def _save_biencoder_outs(save_preds_dir, nns, dists, pred_mention_bounds, cand_scores, mention_scores, runtime):
    np.save(os.path.join(save_preds_dir, "biencoder_nns.npy"), nns)
    np.save(os.path.join(save_preds_dir, "biencoder_dists.npy"), dists)
    np.save(os.path.join(save_preds_dir, "biencoder_mention_bounds.npy"), pred_mention_bounds)
    np.save(os.path.join(save_preds_dir, "biencoder_cand_scores.npy"), cand_scores)
    np.save(os.path.join(save_preds_dir, "biencoder_mention_scores.npy"), mention_scores)
    with open(os.path.join(save_preds_dir, "runtime.txt"), "w") as wf:
        wf.write(str(runtime))


def _load_biencoder_outs(save_preds_dir):
    nns = np.load(os.path.join(save_preds_dir, "biencoder_nns.npy"), allow_pickle=True)
    dists = np.load(os.path.join(save_preds_dir, "biencoder_dists.npy"), allow_pickle=True)
    pred_mention_bounds = np.load(os.path.join(save_preds_dir, "biencoder_mention_bounds.npy"), allow_pickle=True)
    cand_scores = np.load(os.path.join(save_preds_dir, "biencoder_cand_scores.npy"), allow_pickle=True)
    mention_scores = np.load(os.path.join(save_preds_dir, "biencoder_mention_scores.npy"), allow_pickle=True)
    runtime = float(open(os.path.join(args.save_preds_dir, "runtime.txt")).read())
    return nns, dists, pred_mention_bounds, cand_scores, mention_scores, runtime


def display_metrics(
    num_correct, num_predicted, num_gold, prefix="",
):
    p = 0 if num_predicted == 0 else float(num_correct) / float(num_predicted)
    r = 0 if num_gold == 0 else float(num_correct) / float(num_gold)
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    print("{0}precision = {1} / {2} = {3}".format(prefix, num_correct, num_predicted, p))
    print("{0}recall = {1} / {2} = {3}".format(prefix, num_correct, num_gold, r))
    print("{0}f1 = {1}".format(prefix, f1))


def load_models(args, logger):
    # load biencoder model
    if logger: logger.info("Loading biencoder model")
    try:
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(args.biencoder_config) as json_file:
            for line in json_file:
                line = line.replace("'", "\"")
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                biencoder_params = json.loads(line)
                break
    biencoder_params["path_to_model"] = args.biencoder_model
    biencoder_params["cand_token_ids_path"] = args.cand_token_ids_path
    biencoder_params["eval_batch_size"] = getattr(args, 'eval_batch_size', 8)
    biencoder_params["no_cuda"] = (not getattr(args, 'use_cuda', False) or not torch.cuda.is_available())
    if biencoder_params["no_cuda"]:
        biencoder_params["data_parallel"] = False
    biencoder_params["load_cand_enc_only"] = False
    if getattr(args, 'max_context_length', None) is not None:
        biencoder_params["max_context_length"] = args.max_context_length
    biencoder = load_biencoder(biencoder_params)
    if biencoder_params["no_cuda"] and type(biencoder.model).__name__ == 'DataParallel':
        biencoder.model = biencoder.model.module
    elif not biencoder_params["no_cuda"] and type(biencoder.model).__name__ != 'DataParallel':
        biencoder.model = torch.nn.DataParallel(biencoder.model)

    # load candidate entities
    if logger: logger.info("Loading candidate entities")

    (
        candidate_encoding,
        indexer,
        id2title,
        id2text,
        id2wikidata,
    ) = _load_candidates(
        args.entity_catalogue, args.entity_encoding,
        args.faiss_index, args.index_path, logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        indexer,
        id2title,
        id2text,
        id2wikidata,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    candidate_encoding,
    indexer,
    id2title,
    id2text,
    id2wikidata,
    test_data=None,
    sp=None,
):

    if not test_data and not getattr(args, 'test_mentions', None) and not getattr(args, 'interactive', None):
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entities (--test_entities) or manually pass in test data"
        )
        raise ValueError(msg)
    
    if getattr(args, 'save_preds_dir', None) is not None and not os.path.exists(args.save_preds_dir):
        os.makedirs(args.save_preds_dir)
        print("Saving preds in {}".format(args.save_preds_dir))

    stopping_condition = False
    threshold = float(args.threshold)
    if args.threshold_type == "top_entity_by_mention":
        assert args.mention_threshold is not None
        mention_threshold = float(args.mention_threshold)
    else:
        mention_threshold = threshold
    if args.interactive:
        while not stopping_condition:

            if logger: logger.info("interactive mode")

            # Interactive
            text = input("insert text: ")

            # Prepare data
            samples = [{"id": "-1", "text": text}]
            dataloader = _process_biencoder_dataloader(
                samples, biencoder.tokenizer, biencoder_params, logger,
            )

            # Run inference
            nns, dists, pred_mention_bounds, mention_scores, cand_scores = _run_biencoder(
                args, biencoder, dataloader, candidate_encoding, samples=samples,
                num_cand_mentions=args.num_cand_mentions, num_cand_entities=args.num_cand_entities,
                device="cpu" if biencoder_params["no_cuda"] else "cuda",
                threshold=mention_threshold, indexer=indexer,
            )

            action = "c"
            while action == "c":
                all_entity_preds = get_predictions(
                    args, dataloader, biencoder_params,
                    samples, nns, dists, mention_scores, cand_scores,
                    pred_mention_bounds, id2title, threshold=threshold,
                    mention_threshold=mention_threshold,
                )[0]

                pred_triples = all_entity_preds[0]['pred_triples']
                _print_colorful_text(all_entity_preds[0]['tokens'], tokenizer, pred_triples)
                _print_colorful_prediction(all_entity_preds, pred_triples, id2text, id2wikidata)
                action = input("Next question [n] / change threshold [c]: ")
                while action != "n" and action != "c":
                    action = input("Next question [n] / change threshold [c]: ")
                if action == "c":
                    print("Current threshold {}".format(threshold))
                    while True:
                        threshold = input("New threshold (increase for less cands, decrease for more cands): ")
                        try:
                            threshold = float(threshold)
                            break
                        except:
                            print("Error! Expected float, got {}. Try again.".format(threshold))
    
    else:
        if not test_data:
            samples, num_unk = _get_test_samples(
                args.test_mentions, args.test_entities, logger,
            )
        else:
            samples = test_data

        if logger: logger.info("Preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params, None,
        )

        stopping_condition = True

        # prepare the data for biencoder
        # run biencoder if predictions not saved
        if not getattr(args, 'save_preds_dir', None) or not os.path.exists(
                os.path.join(args.save_preds_dir, 'biencoder_mention_bounds.npy')):

            # run biencoder
            if logger: logger.info("Running biencoder...")

            start_time = time.time()
            filtered_embeddings_ctxt, filtered_mention_bounds, filtered_mention_scores, filtered_mask, filtered_tokens,entity_text = _run_biencoder(
                args, biencoder, dataloader, candidate_encoding, samples=samples,
                num_cand_mentions=args.num_cand_mentions, num_cand_entities=args.num_cand_entities,
                device="cpu" if biencoder_params["no_cuda"] else "cuda",
                threshold=mention_threshold, indexer=indexer,
            )
            end_time = time.time()
            if logger: logger.info("Finished running biencoder")

            runtime = end_time - start_time
            
        

        
        all_entity_preds = []
        for idx, sample in enumerate(samples):
            ads_tok_text,ads_tok_enitity_ids,match  = bert_to_ads(sample["text"],entity_text[idx],sp=sp)
            entity_results = {
                "id": sample["id"],
                "text": sample["text"],
                "keyword": sample["keyword"],
                "entityTF": filtered_mask[idx],
                "tokens": filtered_tokens[idx],
                "entity_span" : filtered_mention_bounds[idx],
                "entity": entity_text[idx],
                "score": filtered_mention_scores[idx].tolist(),
                "entity_embedding" : filtered_embeddings_ctxt[idx].tolist(),
                "ads_tokens": ads_tok_text,
                "ads_entity_span":ads_tok_enitity_ids,
                "match": match
            }
            all_entity_preds.append(entity_results)

        print("*--------*")
        print("*--------*")
        print("biencoder runtime = {}".format(runtime))
        print("*--------*")

        return all_entity_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug_biencoder", "-db", action="store_true", default=False, help="Debug biencoder"
    )
    # evaluation mode
    parser.add_argument(
        "--get_predictions", "-p", action="store_true", default=False, help="Getting predictions mode. Does not filter at crossencoder step."
    )
    
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset.",default = "/data/vmorisetty/data/elq_split/elq_test.tsv"
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities.",
        default="/data/vmorisetty/models/elq_models/entity.jsonl",  # ALL WIKIPEDIA!
    )

    parser.add_argument(
        "--save_preds_dir", type=str, help="Directory to save model predictions to."
    )
    parser.add_argument(
        "--mention_threshold", type=str, default=None,
        dest="mention_threshold",
        help="Used if threshold type is `top_entity_by_mention`. "
        "Threshold for mention score, for which mentions will be pruned if they fall under that threshold. "
        "Set to '-inf' to get all mentions."
    )
    parser.add_argument(
        "--threshold", type=str, default="-4.5",
        dest="threshold",
        help="Threshold for final joint score, for which examples will be pruned if they fall under that threshold. "
        "Set to `-inf` to get all entities."
    )
    parser.add_argument(
        "--num_cand_mentions", type=int, default=50, help="Number of mention candidates to consider per example (at most)"
    )
    parser.add_argument(
        "--num_cand_entities", type=int, default=10, help="Number of entity candidates to consider per mention (at most)"
    )
    parser.add_argument(
        "--threshold_type", type=str, default="joint",
        choices=["joint", "top_entity_by_mention"],
        help="How to threshold the final candidates. "
        "`top_entity_by_mention`: get top candidate (with entity score) for each predicted mention bound. "
        "`joint`: by thresholding joint score."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="/data/vmorisetty/models/wiki_ads20M_2epochft.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="/data/vmorisetty/models/finetuned_elqmodels/experiments/elq_train.tsv/all_avg_20_true_true_bert_large_qa_linear/training_params.txt",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--cand_token_ids_path",
        dest="cand_token_ids_path",
        type=str,
        default="/data/vmorisetty/models/elq_models/entity_token_ids_128.t7",  # ALL WIKIPEDIA!
        help="Path to tokenized entity catalogue",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        default="/data/vmorisetty/models/elq_models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        default="/data/vmorisetty/models/elq_models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=8,
        help="Crossencoder's batch size for evaluation",
    )
    parser.add_argument(
        "--faiss_index",
        dest="faiss_index",
        type=str,
        default="hnsw",
        choices=["hnsw", "flat", "ivfflat", "none"],
        help="whether to use faiss index",
    )
    parser.add_argument(
        "--index_path",
        dest="index_path",
        type=str,
        default="/data/vmorisetty/models/elq_models/faiss_hnsw_index.pkl",
        help="path to load indexer",
    )
    parser.add_argument(
        "--max_context_length",
        dest="max_context_length",
        type=int,
        help="Maximum length of context. (Don't set to inherit from training config)",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--use_cuda", dest="use_cuda", action="store_true", default=True, help="run on gpu"
    )
    parser.add_argument(
        "--no_logger", dest="no_logger", action="store_true", default=False, help="don't log progress"
    )


    args = parser.parse_args()

    logger = None
    if not args.no_logger:
        logger = utils.get_logger(args.output_path)
        logger.setLevel(10)

    models = load_models(args, logger)
    run(args, logger, *models)

