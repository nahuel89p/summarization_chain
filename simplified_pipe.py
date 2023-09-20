#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:06:15 2023

@author: nahuelpatino
"""

from pipeutils import *

path='/Users/user/summarizations/'

# Replace 'YOUR_API_KEY' with your actual YouTube Data API v3 key
api_key = ''

#Choose one or many youtube Channels
channel_ids = ['UCKMeK-HGHfUFFArZ91rzv5A']

knwoledge_base={'UCKMeK-HGHfUFFArZ91rzv5A': "The host is usually Adam Taggart"}

trimdict={'some optional video channel id': 19} #seconds to cut if videos have a fixed intro

# Specify the maximum number of videos to fetch
max_results = 1

videos = get_youtube_channel_videos(api_key, channel_id, max_results)

#Gen mp4 file
gen_transcripts(path,videos,False, trimdict) #mp4 file will download

#### cloud bit: ####
# Load the mp4 files in the colab notebook and download the resulting jsons
####################
    
prefix = "transcript_"
to_omit= ['optionalfile.json']
result_dict = load_json_files_with_prefix(path, prefix, to_omit)

###############
import time
import json
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    Prompt,
    PromptHelper,
    LLMPredictor,
    Document,
    LangchainEmbedding
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import gc
import copy
import os
import numpy as np
import pandas as pd

gc.collect()
ytfiles = os.listdir(path)

for key_to_keep in result_dict:
    key_to_keep_processed = "report_"+key_to_keep+".json"
    print(key_to_keep)
    if key_to_keep_processed not in ytfiles:

        new_dict = keep_one_key(result_dict, key_to_keep)
        iv = new_dict[key_to_keep]['text']
        substrings = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02","SPEAKER_03","SPEAKER_04","SPEAKER_05","SPEAKER_06","SPEAKER_07"]
        guestsl= ["SPEAKER_1 says", "SPEAKER_2 says", "SPEAKER_3 says","SPEAKER_4 says","SPEAKER_5 says","SPEAKER_6 says","SPEAKER_7 says","SPEAKER_8 says"]
        
        result = find_first_substring(iv, substrings)                
        
        n_speakers=max([i+1 for i, n in enumerate(substrings) if n in iv])
        if n_speakers == 1:
            iv=iv.replace('SPEAKER_00', 'HOST')
        else:
            nn=0
            for i in substrings:
                result = find_first_substring(iv, substrings)
                if result is not None:
                    iv=iv.replace(result,guestsl[nn])
                    nn+=1

        desc = videos.loc[ videos.Video_id == key_to_keep, 'Description'].values[0]
        vidtitle = videos.loc[ videos.Video_id == key_to_keep, 'Video Title'].values[0]
        chantitle = videos.loc[ videos.Video_id == key_to_keep, 'Channel_title'].values[0]
        chanid = videos.loc[ videos.Video_id == key_to_keep, 'Channel_id'].values[0]
        
        if chanid in knwoledge_base:
            customknowledge= knwoledge_base[chanid]
        else:
            customknowledge="None."
            
################################################    
############ 1) Who are the participants? ############
################################################

        if n_speakers ==1:
            template='''
            Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### Instruction:
            Below is an automatic-generated transcription of a podcast in YouTube, plus the YouTube video description and the YouTube video title. The host is the only speaker and the diarization system assigned him the naive label "HOST". Your task is to find out from the context provided what is the host's name. Think step by step.
            
            ### Input:    
            YouTube video description:
            {desc}
            
            Interview transcription text (warning: names can be mispelled here):    
            {text}
            
            YouTube video title:
            {vidtitle}
            
            YouTube channel name (sometimes the channel is named after the owner of the channel, who might be the host):
            {chantitle}
            
            Additional information:
            {customknowledge}    
            
            ### Response:
            Let's reason step by step.
            '''      
        
            sample=iv[0:1500]
            
            fullprompt = template.format(text=sample, desc = desc, vidtitle = vidtitle, chantitle = chantitle, customknowledge=customknowledge)
            
            local_path =  '/Users/nahuelpatino/Downloads/nous-hermes-llama2-70b.Q4_0.gguf' # Available in HuggingFace
            
            llm = Llama(
                seed= 0,
                use_mmap= False,
                use_mlock=True,
                n_threads=8,
                model_path=local_path,
                n_gpu_layers=80,
                n_ctx=4096,
                n_batch=256,
                f16_kv=True,
                verbose=True,
                mirostat_mode = 1,
                mirostat_tau= 3,
                mirostat_eta = 1.2,
                last_n_tokens_size=300
                )

            output = llm(fullprompt, stop=['[INST]','[/INST]'], max_tokens=500,temperature=0, echo=False,repeat_penalty=1.18)
            
            intel=output['choices'][0]['text']
        
            del llm
            gc.collect()
            
        else:
        
            
            template=""" 
            YouTube video title:
            {vidtitle}
            
            YouTube video description:
            {desc}
            
            YouTube channel name:
            {chantitle}
            
            Addition information:
            {customknowledge} 
            """
            
            contents = template.format(desc = desc, vidtitle = vidtitle, chantitle = chantitle, customknowledge=customknowledge)
            docs= [Document(text=contents)]
            

            #upstage llama instruct
            template = (
                "### System:\n"
                "You are a helpful assistant. Be helpful, brief and concise in your answers.\n"
                "### User:\n"
                "Below is information about a podcast episode in YouTube:\n"
                "{context_str}\n"
                "Given the information above about the podcast episode in YouTube, please answer this question to identify accurately all the participants of the podcast: {query_str} . Let's think step by step using all the information provided.\n"
                "### Assistant:\n"
            )
            
            
            qa_template = Prompt(template)
            local_path =  '/Users/nahuelpatino/Downloads/upstage-llama-2-70b-instruct-v2.Q4_0.gguf' #chosen one
                        
            prompt_helper = PromptHelper(context_window=4096)
            
            # Callbacks support token-wise streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=65, chunk_overlap=15))
            
            llm = LlamaCpp(
                seed= 0,
                temperature=0.3,
                use_mmap= False,
                use_mlock=True,
                n_threads=8,
                model_path=local_path,
                n_gpu_layers=1,
                n_ctx=4096,
                n_batch=256,
                f16_kv=True,
                callback_manager=callback_manager,
                verbose=True,
                )

            embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
            
            llm_predictor = LLMPredictor(llm=llm)
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                                           prompt_helper=prompt_helper,
                                                           embed_model=embed_model, 
                                                           node_parser=node_parser
            )
            
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            
            response = index.as_query_engine( similarity_top_k = 6,
                                             text_qa_template=qa_template).query("Who are the participants of this conversation? And who is the host?")
            
            intel=response.response.strip()
            
            del llm
            del llm_predictor
            del service_context
            del index
            gc.collect()
        
        gc.collect()
        gc.collect()
        gc.collect()
        
################################################    
############ 2) Refine participants statement ############
################################################

        #hermes
        template='''
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
        ### Instruction:
        Below is a statement declaring who are the participants of a conversation.
        Your task is to shorten the statement, keeping only the essential information on who the people are.
            
        ### Input:
        Statement:
        {identifications}
        
        ### Response:
        
        '''   
        
        local_path =  '/Users/nahuelpatino/Downloads/nous-hermes-llama2-13b.gguf.q4_K_M.bin' #chosen one
        fullprompt = template.format( identifications = intel )
        
        llm = Llama(
            seed= 0,
            use_mmap= False,
            use_mlock=True,
            n_threads=8,
            model_path=local_path,
            n_gpu_layers=80,
            n_ctx=4096,
            n_batch=512,
            f16_kv=True, 
            verbose=True
            )
        
        refinement = llm(fullprompt, stop=['[INST]','[/INST]'], max_tokens=200,temperature=0, echo=False)
        
        intel=refinement['choices'][0]['text'].strip()
        
        del llm
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()

################################################    
############ 3) Reason out who is who ############
################################################        
        if n_speakers>1:

            template = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
                "### Input:\n"
                "Podcast snippets:\n"
                "{context_str}\n"
                "### Instruction:\n"
                "The snippets above belong to the same podcast episode and were automatically generated.\n" 
                "The diarization system assigned naive 'SPEAKER_' labels to the participants and your task is to match each 'SPEAKER_' with its real name.\n"
                "We know the particiapnts are {query_str}\n"
                "Keep in mind that two different names can't be assigned to one same 'SPEAKER_'.\n"
                "\n"
                "Also keep in mind that if a speaker says to another speaker 'I'm Jorge. How are you Thelma?', then it can be inferred that the first speaker is Jorge, as he is introducing himeself, and the next speaker to answer is likely Thelma answering Jorge's question.\n"
                "Do not reproduce the snipptes with the names replaced, just determine who is who.\n"
                "\n"
                "### Response:\n"
                "Let's reason step by step."
            )
                
            local_path =  '/Users/nahuelpatino/Downloads/platypus2-70b-instruct.Q4_0.gguf' # Available in HuggingFace
            
            llm = LlamaCpp(
                seed= 0,
                use_mmap= False,
                use_mlock=True,
                n_threads=8,
                model_path=local_path,
                n_gpu_layers=1,
                n_ctx=4096,
                n_batch=512,
                f16_kv=True, 
                callback_manager=callback_manager,
                verbose=True,
                mirostat_mode = 1,
                mirostat_tau= 3,
                mirostat_eta = 1.2,
                max_tokens=600
                )
                        
            docs= gen_llama_docs(iv)
            node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=120, chunk_overlap=20))
            
            qa_template = Prompt(template)
            llm_predictor = LLMPredictor(llm=llm)
            
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                                                           prompt_helper=prompt_helper,
                                                           embed_model=embed_model, 
                                                           node_parser=node_parser)
            
            
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            
            from llama_index.retrievers import BM25Retriever
            
            # BM25 is keyword based, so it's better for this task
            bm25_retriever = BM25Retriever.from_defaults(index, similarity_top_k=6)

            from llama_index.query_engine import RetrieverQueryEngine
            
            query_engine = RetrieverQueryEngine.from_args(
                retriever=bm25_retriever,
                service_context=service_context,
                text_qa_template=qa_template)
                        
            response = query_engine.query(intel)

            intel2=response.response.strip()
            del llm
            del llm_predictor
            del service_context
            del index
            del query_engine
            del bm25_retriever
        else:
            intel2= copy.deepcopy(intel)    
            
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()
        
        
################################################    
############ 4) Create a JSON with the identities ############
################################################           
        if n_speakers ==1:
        #hermes
            template='''
            Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### Instruction:
            Below is a text reasoning what might be the real names the HOST of a podcast. 
            Your task is to return a JSON dictionary mapping the 'HOST' string label with the real name of the host, as per the text suggestion below.
            Important: the keys and values must be enclosed in double quotes.
            Important: the JSON dictionary must follow this pattern:
            json = {{ "Key" : "Value" }}                
            
            ### Input:
            Text:
            {reasoning}
            
            ### Response:
            json = 
            '''   
        else:
            #hermes
            template='''
            Below is an instruction that describes a task. Write a response that appropriately completes the request.
            
            ### Instruction:
            Below is a text reasoning what might be the real names of speakers in an interview that are referred to with a dummy label constructed with the string "SPEAKER_" followed by a number, for example, SPEAKER_1, SPEAKER_2 and so on. 
            Your task is to return a JSON dictionary mapping the SPEAKER_ labels with their real names as per the text suggestion below.
            Important: the JSON dictionary must follow this pattern:
            json = {{ "Key" : "Value" }}                
            
            ### Input:
            Text:
            {reasoning}
            
            ### Response:
            json = 
            '''   
        
        local_path =  '/Users/nahuelpatino/Downloads/nous-hermes-llama2-13b.gguf.q4_K_M.bin' # Available in HuggingFace
        fullprompt = template.format( reasoning = intel2 )
        
        llm = Llama(
            seed= 0,
            use_mmap= False,
            use_mlock=True,
            n_threads=8,
            model_path=local_path,
            n_gpu_layers=80,
            n_ctx=4096,
            n_batch=512,
            f16_kv=True, 
            verbose=True
            )
        
        jsonoutput = llm(fullprompt, stop=['[INST]','[/INST]'], max_tokens=200,temperature=0, echo=False)
        
        jsonoutput = jsonoutput['choices'][0]['text'].strip()
        
        labelsfix=json.loads(jsonoutput)
                
        #labelsfix['Host'] = 'John Doe'
        
        iv_labeled=copy.deepcopy(iv)
        for i in labelsfix:
            iv_labeled = iv_labeled.replace(i,labelsfix[i] )  
        
        del llm
        gc.collect()
        gc.collect()
        gc.collect()
                
        
################################################    
############ 5) Summarize ############
################################################          
        summary_0=''
        summary_1=''
        
        llama2instructtemplate='''
        [INST] <<SYS>>
        You are a professional financial analyst that writes brief summaries for a knowledgable audience. Be technical and concise.
        <<SYS>>
        
        Write a brief summary on the podcast below. Omit non-finance topics and focus on the main idea with regards to financial markets. Explain how an investor should invest according to the idea presented.  
        
        Podcast:
        {text}
        
        Write a brief summary on the podcast above. Omit non-finance topics and focus on the main idea with regards to financial markets. Explain how an investor should invest according to the idea presented.  
        [/INST]
        '''
        llama2instruct32k =  '/Users/nahuelpatino/Downloads/llama-2-7b-32k-instruct.Q8_0.gguf' # Available in HuggingFace
        
        summary = ''
        
        fullprompt = llama2instructtemplate.format(text=iv_labeled)
        
        #try:
        llm = Llama(
            seed= 0,
            use_mmap= False,
            use_mlock=True,
            n_threads=8,
            model_path=llama2instruct32k,
            n_gpu_layers=80,
            n_ctx=4096*8,
            n_batch=512,
            f16_kv=True, # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose=True,
            mirostat_mode = 1,
            mirostat_tau= 3,
            mirostat_eta = 1.2,
            last_n_tokens_size=300
            )
        
        if len(iv_labeled) > 45000:
            middlepos=find_middle_paragraph(iv_labeled)
            fullprompt = llama2instructtemplate.format(text=iv_labeled[0:middlepos])
                
            summary_0=''
            output = llm.create_completion(fullprompt, 
                         stop=['[INST]','[/INST]'], 
                         max_tokens=1000,
                         temperature=0, 
                         repeat_penalty=1.18, 
                         echo=True,
                         stream=True
                         )
            for item in output:
                yielded= item['choices'][0]['text']
                summary_0+= yielded
                
            fullprompt = llama2instructtemplate.format(text=iv_labeled[middlepos:])
            summary_1=''
            output = llm.create_completion(fullprompt, 
                         stop=['[INST]','[/INST]'], 
                         max_tokens=1000,
                         temperature=0, 
                         repeat_penalty=1.18, 
                         echo=True,
                         stream=True
                         )
            for item in output:
                yielded= item['choices'][0]['text']
                summary_1+= yielded
            del llm
            del output
            import gc
            gc.collect()
            gc.collect()
            gc.collect()
        
            llama2instructtemplate = (
                "### System:\n"
                "You are a professional financial analyst that writes reports for a knowledgable audience. Be technical and concise.\n"
                "### User:\n"
                "Below are two summaries for the parts one and two of a podcast, respectively.\n"
                "BEGIN_SUMMARIES\n"
                "\n"
                "Summary of part one:\n"
                "{p1}\n"
                "\n"
                "Summary of part two:\n"
                "{p2}\n"
                "\n"
                "END_SUMMARIES\n"
                "\n"
                "Your task is to rewrite both parts in one coherent, cohesive, concise and professional text. Keep the tone and style, don't add any new information. Do not include trivial topics unrelated to finance and markets.\n"
                "### Assistant:\n"
            )
            
            
            local_path =  '/Users/nahuelpatino/Downloads/upstage-llama-2-70b-instruct-v2.Q4_0.gguf' # Available in HuggingFace
        
            fullprompt = llama2instructtemplate.format(p1=summary_0, p2 = summary_1 )
        
            llm = Llama(
                seed= 0,
                use_mmap= False,
                use_mlock=True,
                n_threads=8,
                model_path=local_path,
                n_gpu_layers=80,
                n_ctx=4096,
                n_batch=512,
                f16_kv=True, 
                verbose=True,
                mirostat_mode = 1,
                mirostat_tau= 3,
                mirostat_eta = 1.2,
                last_n_tokens_size=300
                )
                
            output = llm.create_completion(fullprompt, 
                         stop=['[INST]','[/INST]'], 
                         max_tokens=1000,
                         temperature=0, 
                         repeat_penalty=1.18, 
                         echo=True,
                         stream=True
                         )
            for item in output:
                yielded= item['choices'][0]['text']
                summary+= yielded    
        
        else:
            fullprompt = llama2instructtemplate.format(text=iv_labeled)
                    
            output = llm.create_completion(fullprompt, 
                         stop=['[INST]','[/INST]'], 
                         max_tokens=1000,
                         temperature=0, 
                         repeat_penalty=1.18, 
                         echo=True,
                         stream=True
                         )
            for item in output:
                yielded= item['choices'][0]['text']
                summary+= yielded
                    
        del llm
        gc.collect()
        gc.collect()
        gc.collect()
        
        desc = videos.loc[ videos.Video_id == key_to_keep, 'Description'].values[0]
        chantitle = videos.loc[ videos.Video_id == key_to_keep, 'Channel_title'].values[0]
        Channel_id = videos.loc[ videos.Video_id == key_to_keep, 'Channel_id'].values[0]
        date = videos.loc[ videos.Video_id == key_to_keep, 'Date Uploaded'].values[0]
        url = videos.loc[ videos.Video_id == key_to_keep, 'URL'].values[0]
        ts = pd.to_datetime(str(date)) 
        d = ts.strftime('%Y-%m-%d')
        
        report={
        'Video_id': key_to_keep,
        'Video_title':vidtitle,
        'Insight_1':intel,
        'Insight_2':intel2,
        'Json':labelsfix,
        'Text_labeled': iv_labeled,
        'Summary_p1':summary_0,
        'Summary_p2':summary_1,
        'Summary':summary,
        'Description':desc,
         'Channel_name':chantitle,
         'Channel_id':Channel_id,
         'Date_Uploaded':d,
         'Url':url,
        'Twitted': 'No' }
        
        # Serializing json
        json_report = json.dumps(report, indent=4)
         
        fname=path+'report_'+key_to_keep+'.json'
        # Writing to sample.json
        with open( fname, "w") as outfile:
            outfile.write(json_report)
        

        print(summary_0)
        print(summary_1)
        print(summary)






