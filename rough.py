import numpy
import torch
import io
import json
# print("Entered script")
# candidate_token_ids = torch.load("./models/entity_token_ids_128.t7")
# print("Finished loading")
# print(candidate_token_ids.shape)


samples= {}
with io.open("./models/entity.jsonl", mode="r", encoding="utf-8") as file:
        i = 0
        for line in file:
            data = json.loads(line.strip())
            try:
                samples[data['kb_idx']] = data['text']
                i = i + 1
            except:
                pass
            
            
print(i)
print(samples['Q283637'])