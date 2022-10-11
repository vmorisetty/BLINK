import elq.main_dense2 as elq_dense
import argparse
import sentencepiece as spm
import time

sp = spm.SentencePieceProcessor()
sp.load('sentencepiece.bpe.model')

biencoder_path = "/data/vmorisetty/models/"

models_path = "/data/vmorisetty/models/elq_models/"
output_path = "/data/vmorisetty/data/embeddings.tsv"

config = {
"interactive"        : False,
"models_dir"         : models_path,
"eval_batch_size"    : 32,

# "biencoder_model"    : models_path+"elq_wiki_large.bin",
# "biencoder_config"   : models_path+"elq_large_params.txt",
# "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
# "entity_catalogue"   : models_path+"entity.jsonl",
# "entity_encoding"    : models_path+"all_entities_large.t7",        

"biencoder_model"    : biencoder_path + "wiki_ads20M_2epochft.bin",
"biencoder_config"   : models_path + "elq_large_params.txt",
"cand_token_ids_path": models_path + "entity_token_ids_128.t7",
"entity_catalogue"   : models_path + "entity.jsonl",
"entity_encoding"    : models_path + "all_entities_large.t7", 

"output_path"        : "logs/",                               # logging directory
"faiss_index"        : "hnsw",
"index_path"         : models_path+"faiss_hnsw_index.pkl",
"num_cand_mentions"  : 10,
"num_cand_entities"  : 10,
"threshold_type"     : "joint",
"threshold"          : -4.5,
"use_cuda"           : True,
"n_jobs"             : 12,
"processed_path"     : None
}

args = argparse.Namespace(**config)
print("loading models")
models = elq_dense.load_models(args, logger=None)
print("Finshed loadingm models")

data_list = []
with open("/data/vmorisetty/data/EEM_QK_En_Data.tsv", encoding='utf-8') as f: 
    for i, data in enumerate(f.readlines()):
        l = data.split("\t")
        data_list.append([l[0].strip(),l[1].strip()])
        if i % 100 == 0: 
            print('Completed {}, {}'.format(i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
        if i==1024:
            break
data_to_link = [ {"id": i, "text": data_list[i][0],"keyword":data_list[i][1]} for i in range(len(data_list)) ]
num_inputs = len(data_list)
batch_size = config["eval_batch_size"]
num_batches = num_inputs // batch_size
remainder_batch = num_inputs % batch_size 
print("loaded data correctly")
fw  = open(output_path, "w")

for i in range(num_batches):
    batch_data = data_to_link[i*batch_size: (i+1)*batch_size]
    predictions = elq_dense.run(args, None, *models, test_data=batch_data,sp=sp)
    for prediction in predictions:
        fw.write(str(prediction['id']) + "\t")
        fw.write(prediction['text'] + "\t")
        fw.write(prediction['keyword'] + "\t")
        fw.write(str(prediction['entityTF']) + "\t")
        fw.write(",".join(str(item) for item in prediction['tokens']) + "\t")
        fw.write(",".join(str(item) for item in prediction['entity_span']) + "\t")
        fw.write(str(prediction['score']) + "\t")
        fw.write(" ".join(str(item) for item in prediction['entity_embedding']) + "\t")
        fw.write(",".join(str(item) for item in prediction['ads_tokens']) + "\t")
        fw.write(",".join(str(item) for item in prediction['ads_entity_span']) + "\t")
        fw.write(str(prediction['match']) + "\t")
        fw.write("\n")




                


