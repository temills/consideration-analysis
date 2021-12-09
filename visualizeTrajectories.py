!pip install git+https://github.com/ContextLab/davos.git
import davos
smuggle numpy as np
smuggle pandas as pd
smuggle hypertools as hyp
import os
!git clone https://github.com/temills/item-generation.git
!git clone https://github.com/temills/generation-trajectory.git
!git clone https://github.com/temills/item-descriptor-ratings.git
!pip install pydata-wrangler
import datawrangler as dw
import json
from typing import Dict, Any
import hashlib
import json

with open('subject_generations.json') as f:
    subj_generations = json.load(f)

rating_dict = pd.read_json(os.path.join('item-descriptor-ratings', 'descriptor-ratings/all.json'))

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

#get hash table for subject data
subj_hash = {}
for subj_dict in subj_generations:
  subj_hash[dict_hash(subj_dict)] = subj_dict

#word - embedding mappings based on ratings
word_to_embedding = {}
descriptors = {}
for cat, cat_dict in rating_dict.items():
  word_to_embedding[cat] = {}
  descriptors[cat] = []
  for descriptor, d_dict in cat_dict.items():
    if not (type(d_dict) is dict):
      continue
    descriptors[cat].append(descriptor)
    for word, val in d_dict.items():
      if word == "grizzly bear":
        word = "bear"
      word_to_embedding[cat][word] = word_to_embedding[cat].get(word, [])
      word_to_embedding[cat][word].append(val)

#list of dfs, each containing word embeddings for a subject's responses
def get_embeddings(subj_word_list, cat):
  my_embeddings = []
  count = 0
  for subj_words in subj_word_list:
    subj_embeddings = []
    for word in subj_words:
      if word not in word_to_embedding[cat].keys():
        #this happens for 43 of the 91 subjects... for 61 words total
        embed = [np.mean(list(rating_dict[cat][d].values())) for d in descriptors[cat]]
        #descriptors = 
        #numD = len([i[0] for i in rating_dict[cat].items() if not(pd.isna(i[1]))])
        #embed = [0 for i in range(numD)]
      else:
        embed = word_to_embedding[cat][word]
      subj_embeddings.append(embed)
    my_embeddings.append(pd.DataFrame(subj_embeddings))
  return my_embeddings

def run_on_cat(cat):
  #want list of generations by subject for category, with corresponding list of subject hashes
  cat_generations_list = []
  subj_hashes = []
  for hash, res in subj_hash.items():
    if cat=='animals':
      cat = 'zoo animals' #zoo animals in subj_hash
    if cat=='kitchen':
      cat = 'kitchen appliances'
    if cat == 'restaurants':
      cat = 'chain restaurants'
    #only add if 10 responses
    if cat not in res.keys():
      continue
    if (len(res[cat]) == 10):
      subj_hashes.append(hash)
      cat_generations_list.append(res[cat])
  if cat=='zoo animals':
    cat = 'animals'
  if cat=='kitchen appliances':
    cat = 'kitchen'
  if cat == 'chain restaurants':
    cat = 'restaurants'
  cat_generations = pd.DataFrame(cat_generations_list)
  #cat_generated_words = remove_nones([list(r[1].values) for r in cat_generations.iterrows()])
  cat_generated_words = [list(r[1].values) for r in cat_generations.iterrows()]
  cat_embeddings = get_embeddings(cat_generated_words, cat)
  #cat_embeddings = [x for x in cat_embeddings if x.shape[0] > 0]
  hyp.plot(cat_embeddings, align='SRM', reduce='UMAP')
  raveled_trajectories = np.vstack([x.values.ravel() for x in cat_embeddings])
  hyp.plot(raveled_trajectories, '.', cluster='KMeans', n_clusters=3)
  cluster_labels = hyp.cluster(raveled_trajectories, n_clusters=3)
  c1, c2, c3, h1, h2, h3 = get_clusters(cat_generated_words, subj_hashes, cluster_labels)
  print("cluster 1")
  hyp.plot([x for c, x in zip(cluster_labels, cat_embeddings) if c == 0], align='hyper')
  for l in c1:
    print(l)
  print(h1)
  print("cluster 2")
  hyp.plot([x for c, x in zip(cluster_labels, cat_embeddings) if c == 1], align='hyper')
  for l in c2:
    print(l)
  print(h2)
  print("cluster 3")
  hyp.plot([x for c, x in zip(cluster_labels, cat_embeddings) if c == 2], align='hyper')
  for l in c3:
    print(l)
  print(h3)

run_on_cat('vegetables')