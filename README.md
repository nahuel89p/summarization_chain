# Fintwit Voyager

A summarization chain to process finance podcasts listed in YouTube. 

At the time of writing, [pytorch doesn't support](https://github.com/pytorch/pytorch/issues/77764) some functions for the apple silicon metal performance shaders, so the audio-to-text transcription and diarization must be done in a [google colab notebook](https://github.com/nahuel89p/whisperx_diarization).


Several LLM-mediated steps are introduced to achieve the following:
1) Identify the speakers' names from the youtube video metadata (video title, channel title, video description).
2) Identify the true names of the SPEAKER_x naive labels from the text transcription using a retrieval augmented system.
3) Summarize the final text with the SPEAKER_x labels replaced with their true names.

Instruct based models were used for the steps 1 and 2.
A long context length model was used for the step 3.

https://twitter.com/fintwit_voyager

