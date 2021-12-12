import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import itertools

def correctly_read_csv(fname):
    prep_df = pd.read_csv(fname, converters={"tokens_rep": literal_eval, "tokens": literal_eval, "reference": literal_eval})
    return prep_df

def add_extra_keywords(df, key_list, n_repeat):
    """
    Takes df and keyword list, adds n_repeat occurances of keyword to new column when found in doc
    """
    df['tokens_rep_extra'] = df['tokens_rep']
    for ind in df.index:
        match_list = list()
        for keyword in key_list:
            # Making everything into a bock of text to find broken up keywords
            if keyword in "".join(df['tokens_rep'][ind]):
                match_list.append(keyword)
        if len(match_list) > 0:
            i = 0
            while i < n_repeat:
                df['tokens_rep_extra'][ind] = df['tokens_rep_extra'][ind] + match_list
                i += 1
    return df['tokens_rep_extra']

def only_keywords(df, key_list):
    """
    Will keep only keywords
    Right now is returning some blanks, doesn't work on all data sets
    """
    df['tokens_rep_only'] = ""
    for ind in df.index:
        match_list = list()
        for keyword in key_list:
            # Making everything into a bock of text to find broken up keywords
            if keyword in "".join(df['tokens_rep'][ind]):
                count = 0
                while count < "".join(df['tokens_rep'][ind]).count(keyword):
                    match_list.append(keyword)
                    count += 1
        if len(match_list) > 0:
            df['tokens_rep_only'][ind] = match_list
        else:
            print(df['doc_index'][ind])
            df['tokens_rep_only'][ind] = ['this is bad']
    return df['tokens_rep_only']

def fake_tokenizer(tokens):
    return tokens

def get_all_rep_token_strings(token_list):
    all_rep_token_strings = []
    for d in token_list:
        all_rep_token_strings.append(''.join(d))
    return all_rep_token_strings

def get_key_doc_dict(all_rep_token_strings, doc_ind_list):
    # key word, list of document ids that contain it
    key_doc_dict = {}
    for k in all_key_list:
        k = k.strip()
        if k:
            k_list = []
            for i in range(len(all_rep_token_strings)):
                if k in all_rep_token_strings[i]:
                    k_list.append(doc_ind_list[i])
            if k_list:
                key_doc_dict[k] = k_list
    return key_doc_dict

def get_doc_to_keyword_lst_dict(key_doc_dict):
    # dict with doc_id as key and list of keywords in it
    doc_to_keyword_lst_dict = {}
    for k,lst in key_doc_dict.items():
        for doc_id in lst:
            if doc_id in doc_to_keyword_lst_dict:
                doc_to_keyword_lst_dict[doc_id].append(k)
            else:
                doc_to_keyword_lst_dict[doc_id] = [k]
    return doc_to_keyword_lst_dict

def get_doc_to_docs_with_overlap_dict(doc_to_keyword_lst_dict):
    # get list of lists of overlapping keys per document
    doc_to_docs_with_overlap_dict = {}
    for doc_id, keywords_lst in doc_to_keyword_lst_dict.items():
        keyword_set = set(keywords_lst)
        for doc_id_next, keywords_lst_next in doc_to_keyword_lst_dict.items():
            if doc_id_next != doc_id:
                keyword_set_next = set(keywords_lst_next)
                intersection = keyword_set.intersection(keyword_set_next)
                if intersection:
                    if doc_id in doc_to_docs_with_overlap_dict:
                        doc_to_docs_with_overlap_dict[doc_id][doc_id_next] = intersection
                    else:
                        doc_to_docs_with_overlap_dict[doc_id] = {doc_id_next: intersection}
    return doc_to_docs_with_overlap_dict

def rank_most_similar_documents(input_doc_embedding, all_docs_embeddings):
    """
    Get top n matched documents plus their indexes
    Return list of tuples where first position in each tuple is index and second position is the probability
    """
    distances = cosine_similarity(input_doc_embedding, all_docs_embeddings)[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    res_with_doc_index = []
    return results

def rank_most_similar_documents_docid(input_doc_embedding, all_docs_embeddings, doc_ids_all_embeddings):
    """
    Get top n matched documents plus their indexes
    Return list of tuples where first position in each tuple is index and second position is the probability
    """
    distances = cosine_similarity(input_doc_embedding, all_docs_embeddings)[0]
    results = zip(doc_ids_all_embeddings, distances)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    res_with_doc_index = []
    return results

def rank_all_embeddings(all_embeddings, doc_index_list):
    rank_probs_all = []
    rank_inds_all = []
    for i in range(len(doc_index_list)):
        rank_results = rank_most_similar_documents(all_embeddings[i], all_embeddings)
        rank_probs = []
        rank_inds = []
        for ind,prob in rank_results:
            if ind != i:
                rank_probs.append(prob)
                rank_inds.append(doc_index_list[ind])
        rank_probs_all.append(rank_probs)
        rank_inds_all.append(rank_inds)
    return rank_probs_all, rank_inds_all

def rank_all_keyword_overlap_embeddings(search_df, doc_overlap_dict, all_embeds, id_to_ind):
    rank_probs_all = []
    rank_inds_all = []
    for i in range(len(search_df['doc_index'])):
        # get list of embeddings for just the input document
        if search_df['doc_index'][i] in doc_overlap_dict:
            overlap_inner_dict = doc_overlap_dict[search_df['doc_index'][i]]
            overlap_docs = overlap_inner_dict.keys()
            overlap_embeds = []
            for doc_id in overlap_docs:
                the_embed = all_embeds[id_to_ind[doc_id]].toarray().flatten()
                overlap_embeds.append(the_embed)
            if overlap_embeds:
                overlap_embeds = np.stack(overlap_embeds)
                #print(overlap_embeds.shape)
                rank_results = rank_most_similar_documents_docid(all_embeds[i], overlap_embeds, overlap_docs)
                rank_probs = []
                rank_inds = []
                for d_id,prob in rank_results:
                    rank_probs.append(prob)
                    rank_inds.append(d_id)
                rank_probs_all.append(rank_probs)
                rank_inds_all.append(rank_inds)
            else:
                rank_probs_all.append([])
                rank_inds_all.append([])
        else:
            rank_probs_all.append([])
            rank_inds_all.append([])
    return rank_probs_all, rank_inds_all

def create_guess_output(rank_probs, rank_inds, doc_ids, THRESH):
    """
    Returns a df that contains doc_id and matching reference docs
    THRESH controls threshold of match probability   
    """
    probs_to_save = []
    inds_to_save = []
    for i in range(len(rank_inds)):
        this_ind_probs_to_save = []
        this_ind_inds_to_save = []
        this_ind = 0
        prob = rank_probs[i][this_ind]
        while prob >= THRESH:
            this_ind_probs_to_save.append(prob)
            this_ind_inds_to_save.append(rank_inds[i][this_ind])
            this_ind += 1
            prob = rank_probs[i][this_ind]
        probs_to_save.append(this_ind_probs_to_save)
        inds_to_save.append(this_ind_inds_to_save)

    # Creating output df with index and match
    dict_list = list()
    for i in range(len(doc_ids)):
        if len(inds_to_save[i]) == 0:
            pass
        else:
            for ind in inds_to_save[i]:
                new_row = {'Test' :doc_ids[i], "Reference": ind}
                dict_list.append(new_row)
    output_df = pd.DataFrame(dict_list)
    return output_df

def create_guess_output_neg_match(rank_probs, rank_inds, doc_ids, THRESH, matches_list):
    """
    Returns a df that contains doc_id and matching reference docs
    THRESH controls threshold of match probability   
    """
    probs_to_save = []
    inds_to_save = []
    for i in range(len(rank_inds)):
        this_ind_probs_to_save = []
        this_ind_inds_to_save = []
        this_ind = 0
        prob = rank_probs[i][this_ind]
        while prob >= THRESH:
            this_ind_probs_to_save.append(prob)
            this_ind_inds_to_save.append(rank_inds[i][this_ind])
            this_ind += 1
            prob = rank_probs[i][this_ind]
        probs_to_save.append(this_ind_probs_to_save)
        inds_to_save.append(this_ind_inds_to_save)
    return inds_to_save

    # Creating output df with index and match
    dict_list = list()
    for i in range(len(doc_ids)):
        if len(inds_to_save[i]) == 0:
            pass
        else:
            for ind in inds_to_save[i]:
                new_row = {'Test' :doc_ids[i], "Reference": ind}
                dict_list.append(new_row)
    output_df = pd.DataFrame(dict_list)
    
    return output_df

def check_training_results(training_labels_df, guesses_df):
    # This is stupid but it works
    # Returns f1_measure (AI cup eval metric)
    output_touples = [tuple(r) for r in guesses_df.to_numpy()]
    training_label_touples = [tuple(r) for r in training_labels_df.to_numpy()]
    n_correct_guesses = len(set(output_touples) & set(training_label_touples))
    n_correct_guesses
    n_guesses = len(output_touples)
    n_correct_answers = len(training_label_touples)
    precision = n_correct_guesses / n_guesses
    recall = n_correct_guesses / n_correct_answers
    f1_measure = 2*((precision * recall)/(precision + recall))
    # print("Correct Guesses: {}\nNumber of Guesses: {}\nPrecision: {} \n Recall: {}\n Score: {}".format(n_correct_guesses, n_guesses, precision, recall, f1_measure))
    # print(f1_measure)
    return {"score": f1_measure, "precision": precision, "recall": recall}



def tfidf_cosine_matches(df, tfidf, doc_tokens_col, THRESH):
    """
    Processes df, returns guesses df with given doc_tokens_col, threshold, and tfidf
    """
    
    doc_tokens = df[doc_tokens_col]
    tfidf_embeddings = tfidf.fit_transform(doc_tokens)
    
    doc_ids = df['doc_index']
    rank_probs, rank_inds = rank_all_embeddings(tfidf_embeddings, doc_ids)
    
    # pickle.dump(tfidf, open('tfidf_model.pkl', 'wb'))
    # pickle.dump(tfidf_embeddings, open('tfidf_embeddings.pkl', 'wb'))

    guesses_output = create_guess_output(rank_probs, rank_inds, doc_ids, THRESH)
    return guesses_output

def multiprocess_tfidf(combo, df, training_labels):
    print(combo)
    tfidf = TfidfVectorizer(
        # ngram_range = combo['ngram_range'],
        # max_df = combo['max_df'],
        # min_df = combo['min_df'],
        # binary = combo['binary'],
        # norm = combo['norm'],
        # use_idf = combo['use_idf'],
        # smooth_idf = combo['smooth_idf'],
        # sublinear_tf = combo['sublinear_tf'],
        tokenizer=fake_tokenizer,
        lowercase=False
    )
    try:
        training_guess = tfidf_cosine_matches(df, tfidf, "tokens_rep_extra", THRESH = combo['threshold'])
        results = check_training_results(training_labels, training_guess)
        combo['score'] = results['score']
        combo['precision'] = results['precision']
        combo['recall'] = results['recall']

    except:
        pass
    return combo