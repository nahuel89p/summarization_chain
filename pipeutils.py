#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:46:06 2023

@author: nahuelpatino
"""

#import whisper
#import ffmpeg

from pytube import YouTube
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import requests
import json  
import os
import numpy as np
from llama_index import Document



def find_first_substring(text, substrings):
    first_occurrence = None
    for substring in substrings:
        index = text.find(substring)
        if index != -1 and (first_occurrence is None or index < first_occurrence[1]):
            first_occurrence = (substring, index)

    if first_occurrence is not None:
        return first_occurrence[0]
    else:
        return None

def gen_llama_docs(result_dict, videosdb):
    documents = []
    for i in result_dict:
        print(i)
        trans_text = result_dict[i]['text']
        
        title=videosdb.loc[ videosdb.Video_id == i, 'Video Title' ].values[0]
        desc=videosdb.loc[ videosdb.Video_id ==  i, 'Description' ].values[0]
        date=videosdb.loc[ videosdb.Video_id ==  i, 'Date Uploaded' ].values[0]
        #speaker=videosdb.loc[ videosdb.Video_id ==  video_id, 'Candidate' ].values[0]
        
        date = convert_date_to_month_year(date)

        document = Document(
            text= trans_text,
            metadata={
                "file_name": i,
                #"title": title,
                #"description": desc,
                "date": date,
                #'speaker':speaker
        
            },
            excluded_llm_metadata_keys=['file_name' , 'description' , 'title'],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
    
    
    
        documents.append(document)
    
    return documents
    
def convert_date_to_month_year(input_date):
    try:
        
        if input_date.dtype == np.dtype('datetime64[ns]'):
            input_date = datetime.utcfromtimestamp(input_date.astype(int) * 1e-9)
            
            # Convert to formatted string
            result_string = input_date.strftime('%B %Y')


        else:
            input_date=input_date[:-10]
            # Parse the input date string into a datetime object
            date_obj = datetime.strptime(input_date, '%Y-%m-%d')
    
            # Get the name of the month and year in numbers
            month_name = date_obj.strftime('%B')
            year_number = date_obj.strftime('%Y')
    
            # Combine the month name and year
            result_string = f"{month_name} {year_number}"

        return result_string
    except ValueError as e:
        print(f"Error: {e}")
        return None




def load_json_files_with_prefix(folder_path, prefix, to_omit):
    data_dict = {}
    
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Loop through each file and load JSON data if the title starts with the specified prefix
    for file in files:

        if file.startswith(prefix) and file.endswith('.json') and file not in to_omit:
            videoid=file.split('script_')[1]
            videoid=videoid.split('.')[0]

            file_path = os.path.join(folder_path, file) 
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                strchain=''
                speaker=''
                last_speaker = False
                for z in json_data['word_segments']:
                    resdict= {}
                    word = z['word']
                    if 'speaker' in z:
                        speaker = z['speaker']
                        if last_speaker == speaker:
                            strchain += word + ' '
                        else:
                            strchain += '\n'+ speaker +': ' + word + ' '
                
                        last_speaker = z['speaker']

                    else:
                        pass
                    resdict['text'] = strchain  
                    resdict['video_id'] = videoid 
                    
                    data_dict[videoid]= resdict

    return data_dict




def get_video_details(video_urls,candidates, api_key):
    video_data = []
    
    for url, candidate in zip(video_urls, candidates):
        try:
            # Fetch video details using pytube
            yt = YouTube(url)
            video_title = yt.title
            date_uploaded = yt.publish_date
            channel_id = yt.channel_id
            length = yt.length
            
            # Fetch video description and channel name using YouTube Data API
            video_id = url.split('/')[-1]
            api_url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet'
            response = requests.get(api_url)
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                description = data['items'][0]['snippet']['description']
                channel_name = data['items'][0]['snippet']['channelTitle']
            else:
                description = 'Description not available'
                channel_name = 'Channel name not available'
            
            video_data.append({
                'Video_id':video_id,
                'URL':url,
                'Video Title': video_title,
                'Date Uploaded': date_uploaded,
                'Channel_ID': channel_id,
                'Channel_title': channel_name,
                'Length': length,
                'Description': description,
                'Candidate': candidate
            })
        except Exception as e:
            print(f"Error processing video: {url}. {e}")
            continue
    
    df = pd.DataFrame(video_data)
    return df

from isodate import parse_duration


def get_youtube_channel_videos(api_key, channel_id, max_results=1, duration_threshold=60*10):
    youtube = build('youtube', 'v3', developerKey=api_key)
    # Fetch the channel's uploaded videos
    response = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    print(response)
    playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    # Fetch video details from the playlist
    videos = []
    next_page_token = None
    while len(videos) < max_results:
        playlist_items = youtube.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=min(max_results - len(videos), 50),
            pageToken=next_page_token
        ).execute()
        for item in playlist_items['items']:
            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            description = item['snippet']['description']
            date_uploaded = item['snippet']['publishedAt']
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            channel_name = item['snippet']['channelTitle']
            # Fetch video duration from video details
            video_response = youtube.videos().list(
                part='contentDetails',
                id=video_id
            ).execute()
            duration = video_response['items'][0]['contentDetails']['duration']
            # Check if the video is a YouTube short based on duration
            if parse_duration(duration).total_seconds() <= duration_threshold:
                continue  # Skip YouTube shorts

            thumbnails = video_response['items'][0]['snippet']['thumbnails']
            thumbnail_urls = {key: value['url'] for key, value in thumbnails.items()}

            videos.append({
                'Video_id': video_id,
                'URL': video_url,
                'Video Title': title,
                'Date Uploaded': date_uploaded,
                'Channel_id': channel_id,
                'Channel_title': channel_name,
                'Length': duration,
                'Description': description,
                'Thumbnails': thumbnail_urls  # Include thumbnails
            })

        next_page_token = playlist_items.get('nextPageToken')

        if not next_page_token:
            break

    videos = pd.DataFrame(videos)
    videos['Date Uploaded'] = pd.to_datetime(videos['Date Uploaded'])

    return videos


def remove_short_lines(text, min_length):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if len(line) >= min_length]
    return '\n'.join(filtered_lines)



from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def gen_transcripts(path, df, model, trimdict):
    for i in range(len(df.URL)):
        files = os.listdir(path)

        video_url = df.URL[i]
        video_id = df.Video_id[i]
        channel_id = df.Channel_id[i]

        print("attempting: " + video_id)

        if video_id + '.mp4' not in files and 'transcript_' + video_id + '.json' not in files:
            try:
                # Download the video
                yt = YouTube(video_url)
                stream = yt.streams.filter(only_audio=True).first()
                audio = stream.download(output_path=path, filename=video_id + '.mp4')
                print('mp4 generated')

                if channel_id in trimdict:
                    print("TRIMMIN'")
                    # Trim the video to the first n seconds
                    trim_seconds=int(trimdict[channel_id] )
                    video_duration = yt.length
                    end_time = video_duration - trim_seconds

                    output_video_path = os.path.join(path, video_id + '_trimmed.mp4')
                    ffmpeg_extract_subclip(audio, int(trimdict[channel_id] ),end_time , targetname=output_video_path)
                    
                    os.remove(os.path.join(path, video_id + '.mp4'))
                    os.rename(output_video_path, os.path.join(path, video_id + '.mp4'))
                    #os.remove(audio)  # Remove the original untrimmed audio
                    audio = output_video_path

            except Exception as e:
                print("mp4 fetch failed: " + video_id)
                print(e)
                pass
        else:
            print("mp4 found: " + video_id)
            pass

        if 'transcript_' + video_id + '.json' not in files and video_id + '.mp4' in files and model is not False:
            print('attempt transcription')

            # Perform transcription on the trimmed or untrimmed video
            #result = model.transcribe(audio)

            #out_file = open(path + 'transcript_' + video_id + '.json', "w")
            #json.dump(result, out_file, indent=6)
            print('transcription generated')

        else:
            pass



def keep_one_key(dictionary, key_to_keep):
    return {key_to_keep: dictionary[key_to_keep]}


import subprocess
import os

def mp4_audio_to_wav(input_mp4, output_wav):
    try:
        # Specify the full path to the ffmpeg executable
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg" # Replace with the path from 'brew --prefix ffmpeg'

        # Get the current environment's PATH variable
        env_path = os.environ.get('PATH', '')

        # Add the directory containing ffmpeg to the PATH
        env_path = f"{os.path.dirname(ffmpeg_path)}:{env_path}"

        # Set the modified PATH variable for the subprocess
        env = os.environ.copy()
        env['PATH'] = env_path

        # Set the desired sample rate (16 kHz)
        sample_rate = 16000
        audio_channels = 2

        # Use FFmpeg to extract the audio from the MP4 file, set sample rate to 16 kHz, stereo channels, and save it as WAV
        subprocess.run([ffmpeg_path, "-i", input_mp4, "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(audio_channels), output_wav], env=env)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

def find_middle_paragraph(text):
    # Split the text into paragraphs based on newline characters ('\n')
    paragraphs = text.split('\n')
    
    # Calculate the middle position of the text
    middle_position = len(text) // 2
    
    # Get all the indexes of newline characters
    newline_indexes = [i for i, char in enumerate(text) if char == '\n']
    
    # Find the newline index that is closest to the middle position
    closest_newline_index = min(newline_indexes, key=lambda x: abs(x - middle_position))
    
    # Find the paragraph index that contains the closest newline index
    closest_paragraph_index = sum(1 for idx in newline_indexes if idx < closest_newline_index)
    
    # Return the closest paragraph and its starting position
    return closest_newline_index






def gen_llama_docs(interview):
    documents = []
    
    chunks=interview.split('\n')
    chunks=[i for i in chunks if len(i)>0  ]

    for z,chunk in enumerate(chunks):
        
        speaker, speech = chunk.split(' says:')[0], chunk.split(' says:')[1]
        try:
            speaker2= chunks[z+1].split(' says:')[0]

        except:
            speaker2= 'everyone'
            
        try:
            speaker0= chunks[z-1].split(' says:')[0]
        except:
            speaker0= 'no one'

        document = Document(
            text= speech,
            metadata={
                "Context": "This snippet is from "+speaker+" speaking to "+speaker0
            },
            #excluded_llm_metadata_keys=['file_name' , 'description' , 'title'],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
    
        documents.append(document)
    
    return documents



def split_text_into_tweets(text, max_length=280):
    # Initialize an empty list to store the tweet segments.
    tweet_segments = []

    # Split the text into words.
    words = text.split()

    # Initialize a variable to keep track of the current tweet.
    current_tweet = ""

    # Iterate through the words.
    for word in words:
        # If adding the current word to the current tweet doesn't exceed the character limit, add it.
        if len(current_tweet + " " + word) <= max_length:
            if current_tweet:
                current_tweet += " "  # Add a space if it's not the first word in the tweet.
            current_tweet += word
        else:
            # If adding the word exceeds the character limit, start a new tweet.
            tweet_segments.append(current_tweet)
            current_tweet = word

    # Add the last tweet segment, if any.
    if current_tweet:
        tweet_segments.append(current_tweet)

    return tweet_segments



otemplates=[
"🎧 A new episode by {podcast} has entered the fintwitsphere! Let's attempt a summarization of the main ideas with regards to financial markets, along with the proposed investment thesis (if any): {url}",

"🎙️ Brace yourselves, the latest episode from {podcast} just dropped into the fintwitsphere! Let's dive into the key insights on financial markets and any investment strategies discussed. Check it out: {url}",

"🚀 Fresh from the studio, {podcast} delivers a new episode to the world of finance! Join us as we break down the top takeaways about financial markets and explore potential investment approaches. Listen now: {url}",

"📢 Calling all finance enthusiasts! {podcast} is back with a brand-new episode, and we're here to distill the wisdom on financial markets and investment strategies. Tune in here: {url}",

"🔥 The fintwitsphere is buzzing with the latest episode from {podcast}! Join us in deciphering the key insights about financial markets and any investment hypotheses presented. Catch it here: {url}",

"📊 Ready to level up your financial knowledge? {podcast} is here with a fresh episode, offering insights into financial markets and potential investment ideas. Discover more: {url}",

"🤑 It's podcast time! {podcast} just dropped a new episode, and we're here to unpack the essential insights about financial markets and any investment recommendations. Listen in: {url}",

"💡 Unlock the secrets of the financial world with {podcast}'s latest episode! We're summarizing the key ideas about financial markets and exploring any investment theses. Dive in now: {url}",

"🌐 Join the financial conversation as {podcast} unveils their newest episode! We're breaking down the highlights on financial markets and dissecting proposed investment strategies. Listen here: {url}",

"📢 Hold onto your hats, folks! {podcast} just released a fresh episode, and we're here to decode the insights on financial markets and any investment approaches discussed. Check it out: {url}",

"💰 Your daily dose of financial wisdom is here! {podcast} brings you a new episode, and we're summarizing the main ideas on financial markets and investment thesis, if any. Listen now: {url}"]







