import sys
import random
import pandas as pd
import numpy as np

from models import load_model
from embedding import simple_embedding
import embedding as emb
import preprocessing as pp

# Load the model
model = load_model('best_model')

# encode the input title
#title = sys.argv[1]
title = "Info and details on OpenAi's new GPT-4o update and how to get it!!"

# loading the language models
w2v_stanford = emb.load_stanford_w2v(printouts=False)
w2v_local = emb.get_local_w2v_model(printouts=False)

# creating a dataframe with the title
data = pd.DataFrame({'title': [title], 'interest': [0]})
data = pp.clean_data(data, printouts=False)
# embedding the title
simple_data_emb = simple_embedding(data=data, w2v_stanford=w2v_stanford, w2v_local=w2v_local, agg='mean', printouts=False)

# data as np
to_predict = np.stack(simple_data_emb['simple_embedding'].values, axis=0)

print(*model.predict(to_predict))
