# import elq.candidate_ranking.utils as utils
# from elq.biencoder.data_process import process_mention_data

# import sentencepiece as spm


# sp = spm.SentencePieceProcessor()
# sp.Load("/home/vmorisetty/BLINK/elq/sentencepiece.bpe.model")


# train_samples = utils.read_dataset("elq_test", "/data/vmorisetty/elq_split")

# print("Num train samples:",len(train_samples))



# valid_data, valid_tensor_data, extra_ret_values = process_mention_data(
#         samples=train_samples[:1024],  # use subset of valid data
#         tokenizer=sp,
#         max_context_length=params["max_context_length"],
#         max_cand_length=params["max_cand_length"],
#         context_key=params["context_key"],
#         title_key=params["title_key"],
#         silent=params["silent"],
#         logger=logger,
#         debug=params["debug"],
#         add_mention_bounds=(not args.no_mention_bounds),
#         candidate_token_ids=None,
#         params=params,
#     )

# import sentencepiece as spm
# sp = spm.SentencePieceProcessor()
# sp.Load("/home/vmorisetty/BLINK/elq/sentencepiece.bpe.model")
# print(sp.encode("[CLS2]"))
# print(sp.EncodeAsPieces("[CLS2]"))

# sp.cls_token



import torch


from elq.biencoder.biencoder import BiEncoderRanker

from elq.common.params import ElqParser
params = {'silent': False, 'debug': False, 'data_parallel': True, 'no_cuda': False, 'top_k': 10, 'seed': 52313, 'zeshel': True, 'max_seq_length': 256, 'max_context_length': 128, 'max_cand_length': 128, 'path_to_model': None, 'bert_model': 'bert-large-uncased', 'pull_from_layer': -1, 'lowercase': True, 'context_key': 'context', 'title_key': 'entity', 'out_dim': 1, 'add_linear': False, 'data_path': '/checkpoint/belindali/entity_link/data/tokenized', 'output_path': 'output', 'mention_aggregation_type': 'all_avg', 'no_mention_bounds': True, 'do_mention_detection': True, 'mention_scoring_method': 'qa_linear', 'evaluate': False, 'output_eval_file': None, 'train_batch_size': 32, 'eval_batch_size': 64, 'max_grad_norm': 1.0, 'learning_rate': 1e-05, 'num_train_epochs': 100, 'print_interval': 5, 'eval_interval': 500, 'save_interval': 1, 'warmup_proportion': 0.1, 'gradient_accumulation_steps': 1, 'type_optimization': 'all_encoder_layers', 'shuffle': False, 'start_idx': None, 'end_idx': None, 'last_epoch': 49, 'path_to_trainer_state':None, 'dont_distribute_train_samples': False, 'freeze_cand_enc': True, 'load_cand_enc_only': False, 'cand_enc_path': 'models/all_entities_large.t7', 'index_path': 'models/faiss_hnsw_index.pkl', 'adversarial_training': True, 'get_losses': True,'path_to_model': 'models/elq_wiki_large.bin'}

reranker = BiEncoderRanker(params)
