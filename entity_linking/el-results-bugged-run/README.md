## Legacy Results

This folder contains the Entity Linking results that ran while the temperature chain bug was not yet fixed :( 

The most important runs are the runs where thinking-mode was enabled, but the slug filenames and metadata are wrong:

- The EPG in the candidate generation stage ran at 0.1 temperature
- The reranker and selector LLMs ran at 0.0

The rest of the metadata is still valid.

This folder aims to preserve these results, since inference time for the thinking modes was substantial. Results can be used for brief analysis on F1 performance when thinking is enabled as compared to non-thinking mode.