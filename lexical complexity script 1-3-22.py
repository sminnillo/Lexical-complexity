#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:43:05 2023

@author: sophiaminnillo
"""

#Hello! My name is Sophia Minnillo, and I am a PhD student in Linguistics at UC Davis in the US.
#This is a script that you can use to automatically calculate lexical diversity, density, 
#and sophistication with Spanish language data.
#I created this script to analyze dissertation data from L2 Spanish study abroad students who wrote short emails (~250 words)
#and produced short speech samples (which I transcribed; ~2 minutes in length) in Spanish.

#Part of this script derives from Jenna Frens's (2017) MTLD script (thank you Jenna!):
    #https://github.com/jennafrens/lexical_diversity
#I also use SpaCy as the language model that parsing and POS tagging is based on.
#I learned about how to use SpaCy from Melanie Walsh's (2021) textbook (thank you Melanie):
    #https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/Multilingual/Spanish/03-POS-Keywords-Spanish.html

#%%
#STEP 1: install SpaCy and load packages

#!pip install -U spacy #uncomment and run this if you haven't installed SpaCy before
import spacy
import pandas as pd #for working with data frames
import glob #for importing files
import os #for setting working directory

#%%
#STEP 2: download the SpaCy Spanish language model

#This is the model that was trained on the annotated “AnCora” corpus.
#!python -m spacy download es_core_news_md #uncomment and run this if you haven't before

#load the model and name it 'nlp'
nlp = spacy.load('es_core_news_md')

#%%
#STEP 3: set working directory

#CHANGE WHAT'S IN QUOTES to set it to the name of the file on your computer that you want to access
path="/Users/sophiaminnillo/Desktop/101720 Project" #this is where I have files on my computer
os.chdir(path) #this set's the working directory to whatever you named as the path

#%%
#STEP 4: upload your metadata (information about the participants that helps to label their data)

#CHANGE WHAT'S IN QUOTES to the name of the csv file where you have the metadata
meta = pd.read_csv('sa_metadata_lexis_120622.csv')
#I'm only interested in the naming codes I have for each writing/speech sample, so I just select that column
meta_data = meta[['File_name']]
#Now, I'm turning that column of naming codes into a list. It makes it easier to combine it with other data
meta_data_list = meta_data.values.tolist()    
    
#%%
#STEP 5: upload your data files (writing and transcribed speech samples in my case)

#I set the file path to those data files here
#CHANGE WHAT'S IN QUOTES to match where you have your files,
#but make sure to keep the '/*.txt' at the end
#that allows you to select all of the text files in your directory
#which works great if you have your writing and speech samples in text form
filepath = glob.glob('/Users/sophiaminnillo/Desktop/101720 Project/SA_texts_22/*.txt')   
    
#%%
#STEP 6: read all files and tokenize them (split them into tokens)
#warning: SpaCy includes punctuation as a token, so token count =/ word count

#read all files and tokenize, store in lists of essays then sentences in essay
def readFilesSent(list_names):
    
    list_files = []
    
    for file in list_names:
        a = open(file, 'r', encoding='utf-8') #opening file
        b = a.read() #reading file
        c = nlp(b) #tokenizing, tagging, dependency parsing
    
        list_files.append(c)
    
    return list_files

#list of tokenized files
tokenized_files = readFilesSent(filepath)

#merge with metadata so you know who each text belongs to
tokenized_files_meta = list(zip(meta_data_list, tokenized_files))

#this will be stored in a 'Doc' format, so you can't necessarily look at the data
#but, the tokenized_files variable is important, and will be used in the next steps
    
#%%    
#Brief detour 1: let's look at the tokens

def giveTokens(inputstuff):
    
    list_files = []
    for document in inputstuff:
        docs_stuff = []
        for token in document:
            docs_stuff.append(token.text)
        
        list_files.append(docs_stuff)
            
    return list_files

#list of tokens
token_only = giveTokens(tokenized_files)   

#merge with metadata so you know who each text belongs to
token_only_meta = list(zip(meta_data_list, token_only)) 

#if you want to look at the LEMMAS instead, just delete 'token.text'
#and replace it with 'token.lemma_'
#more about this on SpaCy's website: https://spacy.io/usage/linguistic-features/#_title
    
#%%        
#Brief detour 2: let's look at the tokens and their *POS tags*

def giveTokensandPOS(inputstuff):
    
    list_files = []
    for document in inputstuff:
        docs_stuff = []
        for token in document:
            docs_stuff.append(tuple([token.text, token.pos_]))
        
        list_files.append(docs_stuff)
            
    return list_files

#list of tokens and POS
token_POS = giveTokensandPOS(tokenized_files)

#merge with metadata
token_POS_meta = list(zip(meta_data_list, token_POS))
#create data frame
token_POS_meta_df = pd.DataFrame(token_POS_meta, columns=['text', 'tagged_data'])
#you can export this to csv if you want to
#token_POS_meta_df.to_csv('pos_tagged_SA_123122.csv', index=False)

#if you want to add in analysis of DEPENDENCIES, just add ', token.dep_'
#to the tuple call: 'token.lemma_, token.pos_, token.dep_'
#that could be useful for syntactic complexity analysis

#%%  
#STEP 7: calculate lexical DENSITY
#I am calculating lexical density through the ratio of content words to total words

#First, count how many total words are produced by a student in a text.
def countAllWrds(inputstuff):
    
    list_files = []
    for document in inputstuff:
        num_wrds = 0
        for token in document:
            #this gets rid of non-words that have been marked as tokens
            #(spaces and punctuation marks)
            if token.pos_ != "SPACE" and token.pos_ != "PUNCT":
                num_wrds += 1

    
        list_files.append(num_wrds)
        
    return list_files

#run the function
num_all_words = countAllWrds(tokenized_files)
#merge with metadata
num_all_words_meta = list(zip(meta_data_list, num_all_words))
#create df
num_all_words_meta_df = pd.DataFrame(num_all_words_meta, columns=['text', 'total_count'])

#%%
#still on STEP 7: calculate lexical DENSITY

#number of content words
def countContentWrds(inputstuff):
    
    list_files = []
    for document in inputstuff:
        num_wrds = 0
        for token in document:
            #I am counting verbs, nouns, adjectives, and adverbs as content words
            if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "ADV":
                num_wrds += 1

    
        list_files.append(num_wrds)
        
    return list_files

#run function
num_content_words = countContentWrds(tokenized_files)
#merge with metadata
num_content_words_meta = list(zip(meta_data_list, num_content_words))
#create df
num_content_words_meta_df = pd.DataFrame(num_content_words_meta, columns=['text', 'content_count'])
#%%
#still on STEP 7: calculate lexical DENSITY

#combine 2 dfs, then get ratio of content to total

#adding one column of the num_all_words_meta_df to num_content_words_meta_df
num_content_words_meta_df['total_words'] = num_all_words_meta_df['total_count']

#now divide number of content words by the total number of words
num_content_words_meta_df['lexical_density'] = num_content_words_meta_df['content_count'] / num_content_words_meta_df['total_words']

#congrats, you now have the lexical density of each text!

#Export to csv by changing what's in quotes below:
#num_content_words_meta_df.to_csv('lexical_density_SA_123122.csv', index=False)

#%%
#STEP 8: calculate lexical DIVERSITY: MTLD

#this is Jenna Frens' (2017) script:

import string

# Global trandform for removing punctuation from words
remove_punctuation = str.maketrans('', '', string.punctuation)

# MTLD internal implementation
def mtld_calc(word_array, ttr_threshold):
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0
    
    for token in word_array:
        token = token.translate(remove_punctuation).lower() # trim punctuation, make lowercase
        token_count += 1
        if token not in types:
            type_count +=1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0
    
    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1

# MTLD implementation
def mtld(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 50:
        raise ValueError("Input word list should be at least 50 in length")
    return (mtld_calc(word_array, ttr_threshold) + mtld_calc(word_array[::-1], ttr_threshold)) / 2


# HD-D internals

# x! = x(x-1)(x-2)...(1)
def factorial(x):
    if x <= 1:
        return 1
    else:
        return x * factorial(x - 1)

# n choose r = n(n-1)(n-2)...(n-r+1)/(r!)
def combination(n, r):
    r_fact = factorial(r)
    numerator = 1.0
    num = n-r+1.0
    while num < n+1.0:
        numerator *= num
        num += 1.0
    return numerator / r_fact

# hypergeometric probability: the probability that an n-trial hypergeometric experiment results 
#  in exactly x successes, when the population consists of N items, k of which are classified as successes.
#  (here, population = N, population_successes = k, sample = n, sample_successes = x)
#  h(x; N, n, k) = [ kCx ] * [ N-kCn-x ] / [ NCn ]
def hypergeometric(population, population_successes, sample, sample_successes):
    return (combination(population_successes, sample_successes) *\
            combination(population - population_successes, sample - sample_successes)) /\
            combination(population, sample)
    
# HD-D implementation
def hdd(word_array, sample_size=42.0):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 50:
        raise ValueError("Input word list should be at least 50 in length")

    # Create a dictionary of counts for each type
    type_counts = {}
    for token in word_array:
        token = token.translate(remove_punctuation).lower() # trim punctuation, make lowercase
        if token in type_counts:
            type_counts[token] += 1.0
        else:
            type_counts[token] = 1.0
    # Sum the contribution of each token - "If the sample size is 42, the mean contribution of any given
    #  type is 1/42 multiplied by the percentage of combinations in which the type would be found." (McCarthy & Jarvis 2010)
    hdd_value = 0.0
    for token_type in type_counts.keys():
        contribution = (1.0 - hypergeometric(len(word_array), sample_size, type_counts[token_type], 0.0)) / sample_size
        hdd_value += contribution

    return hdd_value
#%%
#still on STEP 8: calculate lexical DIVERSITY: MTLD

#this runs mltd

list_mtld = []

for file in filepath:
    #particular_file = []
    a = open(file, 'r', encoding='utf-8') #opening file
    b = a.read() #reading file
    c = mtld(b.split())
    list_mtld.append(c)

#merge with metadata
list_mtld_meta = list(zip(meta_data_list, list_mtld))
#create df
list_mtld_meta_df = pd.DataFrame(list_mtld_meta, columns=['text', 'mtld'])

#congrats, you now have the lexical diversity of each text (MTLD)!

#Export to csv by changing what's in quotes below:
#list_mtld_meta_df.to_csv('mtld_SA_120622.csv', index=False)

#%%
#STEP 9: calculate lexical SOPHISTICATION

#As calculating lexical sophistication requires you to extract frequency
#data for the tokens produced from a corpus (I used EsPal: Duchon et al., 2013),
#I am just showing you how to extract all of the CONTENT WORDS in this script
#I have another script in which I upload the frequency data from EsPal,
#match it with the tokens produced, and then average the frequency per text
#as a corpus-based measure of lexical sophistication.

#CONTENT WORDS
#let's gather all of the nouns, adjectives, adv, and verbs in the same df
def giveContentWrds(inputstuff):
    
    list_files = []
    for document in inputstuff:
        docs_stuff = []
        for token in document:
            if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "ADV":   
                docs_stuff.append(token.text) #no POS, just word
                #docs_stuff.append(tuple([token.text, token.pos_])) #use this instead if you want to see the POS tag
                #Also, you can edit what's in the 'if' statement to extract different parts of speech if you'd like.  
        
        list_files.append(docs_stuff)
            
    return list_files

#run function
content_words = giveContentWrds(tokenized_files)
#merge with metadata
content_words_meta = list(zip(meta_data_list, content_words))
#create df
content_words_meta_df = pd.DataFrame(content_words_meta, columns=['text', 'tagged_data'])
#congrats, you now have all of the content words produced for each text
#I would recommend double checking it, because the SpaCy tagger is not always accurate. 
#also, you may want to make all of the tokens lowercase to merge with the corpus-based frequency data

#Export to csv by changing what's in quotes below:
#content_words_meta_df.to_csv('content_words_SA_123122.csv', index=False)

#That's it! Feel free to email me at smminnillo [at] ucdavis.edu if you have any questions.
