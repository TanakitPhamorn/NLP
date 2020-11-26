import stanza as nlp

import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import defaultdict, Counter
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# turns .tsv file into list of lists
def tsv2mat(fname) :
  with open(fname) as f:
     wss = csv.reader(f, delimiter='\t')
     return list(wss)

class Data :
  '''
  builds dataset from dependency edges in .tsv file associating
  <from,link,to> edges and sentences in which they occur;
  links are of the form POS_deprelPOS wuth POS and deprel
  tags concatenated
  '''
  def __init__(self,fname='texts/english') :
    wss = tsv2mat("out/"+fname+".tsv")
    self.sents=tsv2mat("out/"+fname+"_sents.tsv")
    occs=defaultdict(set)
    sids=set()
    for f,r,t,id in wss:
      id=int(id)
      occs[(f,r,t)].add(id)
      sids.add(id)
    self.occs=occs # dict where edges occur