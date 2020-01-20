import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import nltk
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import glob
from collections import defaultdict
from tqdm import tqdm
import json

#import warnings
#warnings.filterwarnings("ignore")




# make dataframe holding disagreements

DATA_ROOT = "/Users/rpryzant/metascience/eLife/data/"


df = pd.read_csv(DATA_ROOT + "kyle/tolabel.csv", sep="|")
df = df[["Manuscript no.", "Reviewer ID", "CleanedComments", "Rec", "Suitable", "ShouldBe", "HumanLabel"]]
df = df.set_index(["Manuscript no."])
scored_bert = pd.read_csv(DATA_ROOT + "kyle/eval_results_full_allelife.txt", 
                          sep="\t", names=["id", "score", "dummy", "text"])


list(scored_bert.sort_values(by="score", ascending=False).iloc[1:10,]["text"])
df["score"] = list(scored_bert.score)
df["Text"] = list(scored_bert.text)




e = pd.read_csv(DATA_ROOT + "raw/eLife_Paper_history_2019_03_15.csv")
e["Manuscript no."] = e["ms"]
e = e.set_index(["Manuscript no."])
e = e.dropna(subset=["full_decision"])

# to get finaldecision, take last non-NA decision of the ones listed here
# note that this excludes rejected by initial decision
e["FinalDecision"] = e.apply(lambda x: list(x[["full_decision", "rev1_decision", "rev2_decision", "rev3_decision", "rev4_decision"]].dropna())[-1], axis=1)
e["outcome"] = np.where(e["FinalDecision"] == "Accept Full Submission", 1, 0)

df_e = df.join(e)



# get disagreements

df_e["review_outcome"] = "none"
df_e["zscore"] = (df_e.score - np.mean(df_e.score))/np.std(df_e.score)
df_e.loc[(df_e.zscore > 1) & (df_e.outcome == 1), "review_outcome" ] = "pos_pos"
df_e.loc[(df_e.zscore > 1) & (df_e.outcome == 0), "review_outcome" ] = "pos_neg"
df_e.loc[(df_e.zscore < -1) & (df_e.outcome == 0), "review_outcome" ] = "neg_neg"
df_e.loc[(df_e.zscore < -1) & (df_e.outcome == 1), "review_outcome" ] = "neg_pos"

# papers with disagreement
disagreement = df_e.loc[(df_e.review_outcome == "pos_neg") | (df_e.review_outcome == "neg_pos")]
disagreement_papers = df_e.loc[set(disagreement.index)]

print(len(disagreement))
print(len(disagreement_papers))


# load initial and reviewer consultations


def get_df(path_glob, sep="|"):
    out = None
    for file in glob.glob(path_glob):
        tmp = pd.read_csv(file, error_bad_lines=False, sep=sep) 
        if out is None:
            out = tmp
        else:
            out = out.append(tmp)
    out = out.set_index(["Manuscript no."])
    return out

initial_consults = get_df(
    DATA_ROOT + "raw/eLife_Initial_Consultation_*_PipeDelimited.txt")

review_consults = get_df(
    DATA_ROOT + "raw/eLife_Reviewer_Consultation_*_PipeDelimited.txt")

reviewers = get_df(DATA_ROOT + "raw/eLife_Reviewers.csv", sep=",")
authors = get_df(DATA_ROOT + "raw/eLife_Authors.csv", sep=",")
authors_orcids = get_df(DATA_ROOT + "raw/eLife_Authors_ORCIDS.csv", sep=",")

author_gender = {}
for i, l in enumerate(open(DATA_ROOT + "kyle/gender_reviewers.csv")):
    if i == 0: continue
    parts = l.strip().split(',')
    author_gender[parts[3]] = parts[-1]

# get into json with key is paper id and value is
# - paper id, outcome, author info
# - reviewer info
# - official reviews, initial consults, review consults
# author info is id, name, gender, institution, city

papers_dict = {}
for paper_id in set(disagreement_papers.index):
    # Skip if only 1 review
    if isinstance(disagreement_papers.loc[paper_id], pd.core.series.Series):
        continue

    papers_dict[paper_id] = {}

    def get_reviews():
        out = []
        
        for _, review in disagreement_papers.loc[paper_id].iterrows():
            tmp = {}
            tmp["text"] = review["CleanedComments"]
            tmp["score"] = review["score"]
            tmp["outcome"] = review["review_outcome"]
            tmp["author"] = str(int(review["Reviewer ID"]))
            out.append(tmp)
        return out
    
    def get_consults(df):
        out = []

        if paper_id not in df.index:
            return out

        # why is this happening
        if isinstance(df.loc[paper_id], pd.core.series.Series):
            return 

        # already sorted by time :)
        for _, review in df.loc[paper_id].iterrows():
            tmp = {}
            tmp["text"] = review["Comment text"]
            tmp["date"] = review["Comment date"]
            tmp["author"] = str(int(review["Commenter ID"]))
            out.append(tmp)
        return out

    def is_accept():
        if isinstance(disagreement_papers.loc[paper_id]["FinalDecision"], str):
            s = disagreement_papers.loc[paper_id]["FinalDecision"]
        else:
            s = disagreement_papers.loc[paper_id]["FinalDecision"].iloc[0]
        return 'Accept' in s

    def get_reviewers(reviews, target_outcome):
        out = []

        for review in reviews:
            if review["outcome"] == target_outcome:
                out.append(review["author"])
        return out

    papers_dict[paper_id]["reviews"] = get_reviews()
    papers_dict[paper_id]["initial_consults"] = get_consults(initial_consults)
    papers_dict[paper_id]["review_consults"] = get_consults(review_consults)

    papers_dict[paper_id]['is_accept'] = is_accept()
    papers_dict[paper_id]['authors'] = [str(int(x)) for x in list(authors.loc[3]["Author ID"])]
    papers_dict[paper_id]['winning_reviewers'] = \
        get_reviewers(papers_dict[paper_id]["reviews"], "neg_neg") + \
        get_reviewers(papers_dict[paper_id]["reviews"], "pos_pos")
    papers_dict[paper_id]['losing_reviewers'] = \
        get_reviewers(papers_dict[paper_id]["reviews"], "pos_neg") + \
        get_reviewers(papers_dict[paper_id]["reviews"], "neg_pos")
    papers_dict[paper_id]['neutral_reviewers'] = get_reviewers(papers_dict[paper_id]["reviews"], "none")



# get author metadata together...this takes a few minutes

def add_authors(d, df, role):

    for pid, row in df.iterrows():
        # every csv has person id as 3rd column, and manuscript id is not index
        try:
            person_id = int(row.iloc[1])
        except ValueError:
            continue
        pid = str(pid)
        if "pids" not in d[person_id]:
            d[person_id]["pids"] = [pid]
        else:
            d[person_id]["pids"] += [pid]

        if "roles" not in d[person_id]:
            d[person_id]["roles"] = [role]
        else:
            d[person_id]["roles"] += [role]

        d[person_id]['gender'] = author_gender.get(str(person_id), 'unk')
        
        # regexes because author/reviewer csvs are different
        d[person_id]["name"] = row.filter(regex=(".*name")).iloc[0]
        d[person_id]["email"] = row.filter(regex=(".*email")).iloc[0]
        d[person_id]["institution"] = row.filter(regex=(".*[Ii]nstitution")).iloc[0]
        
        city_series = row.filter(regex=("[Cc]ity"))
        if not city_series.empty:
            d[person_id]["city"] = city_series.iloc[0]

        country_series = row.filter(regex=("[Cc]ity"))
        if not country_series.empty:
            d[person_id]["country"] = country_series.iloc[0]

        
people_dict = defaultdict(dict)  
add_authors(people_dict, reviewers, "reviewer")
add_authors(people_dict, authors, "author")
add_authors(people_dict, authors_orcids, "author_orchid")


print(len(people_dict))
print(len(papers_dict))


with open('/Users/rpryzant/metascience/eLife/data/processed/papers.json', 'w') as fp:
    json.dump(papers_dict, fp)

with open('/Users/rpryzant/metascience/eLife/data/processed/people.json', 'w') as fp:
    json.dump(people_dict, fp)
    