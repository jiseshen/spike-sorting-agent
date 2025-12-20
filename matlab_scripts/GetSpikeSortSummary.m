function [spikes,spikeSortSummary] = GetSpikeSortSummary(spikes)
%
%
% shreesh@stanford.edu
% 9.10.09
%

reverseSortInd = spikes.reverseSortInd;
spikes.assignments = spikes.hierarchy.assigns;
spikes.origAssignments = spikes.assignments(reverseSortInd);
allclasses = unique(spikes.assignments);
for k=allclasses'
    %fprintf('%d\n',k);
    find(spikes.assignments==k);
    kind=[]; kind = find(spikes.assignments==k);
    if length(kind)>1
        %size(nanmean(spikes.waveforms(kind,:)))
        meanWaves(k+1,:)    = nanmean(spikes.waveforms(kind,:));
        stdWaves(k+1,:)     = nanstd(spikes.waveforms(kind,:));
        seWaves(k+1,:)      = stdWaves(k+1,:)/sqrt(length(kind));
        upWaves(k+1,:)      = meanWaves(k+1,:) + seWaves(k+1,:);
        downWaves(k+1,:)    = meanWaves(k+1,:) - seWaves(k+1,:);
        occupancy(k+1)      = length(kind);
        indices{k+1}        = kind;
    elseif length(kind)==1
        fprintf('hai ');
        %size(spikes.waveforms(kind,:))
        meanWaves(k+1,:)    = spikes.waveforms(kind,:);
        stdWaves(k+1,:)     = nan*ones(size(meanWaves(k+1,:)));
        seWaves(k+1,:)      = stdWaves(k+1,:);
        upWaves(k+1,:)      = meanWaves(k+1,:) + seWaves(k+1,:);
        downWaves(k+1,:)    = meanWaves(k+1,:) - seWaves(k+1,:);
        occupancy(k+1)      = length(kind);
        indices{k+1}        = kind;
    end
end
spikeSortSummary.meanWaves  = meanWaves;
spikeSortSummary.stdWaves   = stdWaves;
spikeSortSummary.seWaves    = seWaves;
spikeSortSummary.upWaves    = upWaves;
spikeSortSummary.downWaves  = downWaves;
spikeSortSummary.occupancy  = occupancy;
spikeSortSummary.indices    = indices;

%% added 6.9.2010 (to enable quick and dirty view of mean waves and occupancy, later on)
spikes.meanWaves  = meanWaves;
spikes.occupancy  = occupancy;



