VerifySpikeSorting_jb.m is the script that can be used to facilitate the curation process. It will call each of the other scripts in this process.

To run it, simply select a spikes.mat file that has been sorted. Once the script has done some preliminary processes, in the command window, there will be a prompt to enter in curation commands.

Commands:

'm c1 c2': merge cluster c1 and cluster c2 (replace c1 and c2 with cluster number)'
's c1': split cluster c1
'p': plot auxillary plots
'w': finish curation, returns curated 'spikes' variable into workspace