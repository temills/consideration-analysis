#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd

#parses question responses, gives dictionaries holding dataframes as described in parse

with open('question_responses.json') as f:
  data = json.load(f)

def parse():
    #for each question, list of dataframes, one per subject, containing each consideration given by subject
    consideration_tables = {}
    #for each question, one dataframe with one row per subject, with one column containing the response given by the subject
    response_tables = {}

    #For a given question, kth consideration dataframe in list and kth row in response datframe correspond to the same subject
    #so for kth subject, consideration_tables[q][k] and response_tables.loc(k,1)

    for q in data.keys():
        consideration_tables[q] = []
        responseList = []
        for subject in data[q]:
            consideration_tables[q].append(pd.DataFrame(subject['considerations'], columns=['Considerations']))
            responseList.append(subject['response'])
        response_tables[q] = pd.DataFrame(responseList, columns=['Responses'])
    
    return consideration_tables, response_tables

parse()