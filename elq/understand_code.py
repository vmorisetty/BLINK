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

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/home/vmorisetty/BLINK/elq/sentencepiece.bpe.model")
print(sp.encode("[CLS2]"))
print(sp.EncodeAsPieces("[CLS2]"))

sp.cls_token