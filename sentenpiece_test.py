import sentencepiece as spm
import time
from transformers import BertTokenizer

sp = spm.SentencePieceProcessor()
sp.load('sentencepiece.bpe.model')




tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "I use nike running shoes to run in the morning"
Entity = (3,5)


text_tokenized_ids = tokenizer.encode(text)

# Entity_text = tokenizer.decode(text_tokenized_ids[Entity[0]+1:Entity[1]+2])

# print(Entity_text)


# print(sp.encode_as_ids(text))
# text_new_ids = sp.encode_as_ids(text)
# print(sp.encode_as_pieces(text))

# print(sp.encode_as_ids(Entity_text))
# entity_new_ids = sp.encode_as_ids(Entity_text)

# print(sp.encode_as_pieces(Entity_text))

# text_len = len(sp.encode_as_ids(text))
# entity_len = len(sp.encode_as_ids(Entity_text))

# entity_new_pos = (0,0)
# match = False
# for i in range(text_len):
#     if text_new_ids[i:i+entity_len] == entity_new_ids:
#         match = True
#         entity_new_pos = (i,i+entity_len-1)
#         break
#print(entity_new_pos)


def bert_to_ads(text,Entity_text,entity_pos,sp):
    #Entity_text = bert_tok.decode(filtered_ids[entity_pos[0]:entity_pos[1]+1])
    text_new_ids = sp.encode_as_ids(text)
    ads_tok_text = sp.encode_as_pieces(text)
    entity_new_ids = sp.encode_as_ids(Entity_text)
    text_len = len(text_new_ids)
    entity_len = len(entity_new_ids)
    ads_tok_enitity_ids = (0,0)
    match = False
    for i in range(text_len):
        if text_new_ids[i:i+entity_len] == entity_new_ids:
            match = True
            ads_tok_enitity_ids = (i,i+entity_len-1)
            break
    return ads_tok_text,ads_tok_enitity_ids,match


t1 = time.time()
for i in range(1000):
    bert_to_ads(text,text_tokenized_ids,Entity,sp,tokenizer)
print("amount encoded in 1 second is:",1000/(time.time()-t1))


# tok_ids = tokenizer.tokenize(text)
# print(tok_ids)
#print("bert decoding:",tokenizer.decode(tok_ids[5:7]))
#print("sp encoding:",sp.encode_as_ids)