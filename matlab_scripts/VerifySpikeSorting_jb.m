function spikes = VerifySpikeSorting_jb(spikesfile,starting_pt)
%
% 9.10.09 shreesh@stanford.edu
%
% 12.13.24 jbedke1@jh.edu - cleaned up script and ensured it is compatible
% with my pipeline

% syntax: VerifySpikeSporting('uStim','Agf#_tr#');
%
% for siteIds, see FinalizeSpikeSorting_uStim.m;
%         'Agf4_tr46','Agf4_tr83',...
%         'Agf6_tr64','Agf6_tr74',...
%         'Agf7_tr20','Agf7_tr25','Agf7_tr45','Agf7_tr50',...
%         'Agf8_tr23','Agf8_tr39','Agf8_tr64','Agf8_tr72','Agf8_tr82','Agf8_tr92','Agf8_tr99',...
%         'Agf10_tr6','Agf10_tr20','Agf10_tr28',...
%         'Agf11_tr28','Agf11_tr35',...
%         'Agf12_tr19',...

%% (getting rid of outlier clusters)
    load(spikesfile,'spikes')
    
    if starting_pt == 1
        spikes.assignments = spikes.hierarchy.assigns;
        spikes.spiketimes = spikes.spiketimes/1000;
    end
    spClasses = unique(spikes.assignments);
    % occupancy = spikes.occupancy;
    for k=1:length(spClasses)
        ind = find(spikes.assignments==spClasses(k));
        occupancy(k,1) = length(ind);
        meanWaves(k,:) = mean(spikes.waveforms(ind,:));
    end
    fprintf('%d ',occupancy); fprintf('\n');
    nSp = size(meanWaves,1);
    
    
    figure; subplot(2,2,1); plot(meanWaves');  title([num2str(nSp)]);
    badClassInd=[];
    % 
    %-- based on max height of waveform and occupancy
    for k=1:length(spClasses)
        maxx(k) = max(abs(meanWaves(k,:)));
        if (maxx(k) >= 500 || occupancy(k)<500), badClassInd = [badClassInd k]; end
    end
    % 
    % % %-- based on occupancy and class number
    % badClasses = find(occupancy<200 | spClasses>=30);
    % 
    mergeOutlier=[];
    mergeOutlier = spClasses(badClassInd);
    for k=1:length(mergeOutlier)
        spikes = merge_clusters(spikes, 0,mergeOutlier(k));
    end
    spClasses = unique(spikes.assignments);
    
    subplot(2,2,2); plot(meanWaves(setdiff(1:nSp,badClassInd),:)'); title([num2str(nSp-length(badClassInd))]);
    pause(0.5);
    fprintf('Pause here (line 79, VerifySpikeSorting.m)\n') ;
    close;
    %%
    
    % PreprocessExperiments(expt{1});
    loc_PlotSpikeSortingStuff(spikes);
    
    fprintf('%d ',unique(spikes.hierarchy.assigns));
    % fprintf('%d ',unique(spikes.occupancy));
    
    doWhat = input('MergeOrSplit [#] ');
    %%%%%%%%%% MODIFIED 4.26.2012:
    while ~strcmp(doWhat(1),'x') & ~strcmp(doWhat(1),'w')
        close all
        if iscell(doWhat),
            doWhatStr=doWhat;clear doWhat;
            for k=1:length(doWhatStr)
                doWhat=doWhatStr{k};
                inps=str2num(doWhat(2:end));
                if strcmp(doWhat(1),'m');
                    spikes = merge_clusters(spikes, inps(1),inps(2));
                elseif strcmp(doWhat(1),'s');
                    spikes = split_cluster(spikes, inps(1));
                elseif strcmp(doWhat(1),'r');
                    inps
                    doWhat
                    spikes = rename_clusters(spikes, inps(2),inps(1));
                elseif strcmp(doWhat(1),'p')
                    plot_supplements(spikes)
                end
            end
        else
            inps=str2num(doWhat(2:end));
            if strcmp(doWhat(1),'m');
                spikes = merge_clusters(spikes, inps(1),inps(2));
            elseif strcmp(doWhat(1),'s');
                spikes = split_cluster(spikes, inps(1));
            elseif strcmp(doWhat(1),'r');
                inps
                doWhat
                spikes = rename_clusters(spikes, inps(2),inps(1));
            end
        end
        spikes = loc_PlotSpikeSortingStuff(spikes);
        fprintf('%d ',unique(spikes.hierarchy.assigns));
        doWhat = input('MergeOrSplit [#] ');
        fprintf('\n\n');
    end %while
end
%%
function spikes = loc_PlotSpikeSortingStuff(spikes)
fig = figure;
jjetf = colormap(jet);
clustlist = unique(spikes.hierarchy.assigns);
colorstep=floor(length(jjetf)/length(clustlist));
jjetm=jjetf(1:colorstep:end,:); %let's get as much separation between colors as possible

for j = 1:length(clustlist)
    k = clustlist(j);
    if k == 0
    else
        spikes.overcluster.colors(k,:)=jjetm(j,:);
    end
end
close(fig)
% fprintf('FIGURE3\n');
spikes = showclust(spikes); %fig5

% fprintf('FIGURE4\n');
scrn_sz = get(0,'ScreenSize');
figure('Position',[1+scrn_sz(3)/2 1 scrn_sz(3)/2 scrn_sz(4)/2])
plot_individual_clus(spikes,spikes.hierarchy.assigns)

% fprintf('FIGURE5\n');
% NewFigure(figW,figH);
figure
aggtree(spikes); title('Aggregation Tree'); %fig7

% fprintf('FIGURE7\n');
% subplot(2,2,4);
%scrsz = get(0,'ScreenSize'); ssfac1=0.4; ssfac2=0.6;
%set(gcf,'Position',[1 ssfac2*scrsz(4) ssfac1*scrsz(3) ssfac2*scrsz(4)]);



end

function plot_supplements(spikes)
jjetf = colormap(jet);
clustlist = unique(spikes.hierarchy.assigns);
colorstep=floor(length(jjetf)/length(clustlist));
jjetm=jjetf(1:colorstep:end,:); %let's get as much separation between colors as possible

for j = 1:length(clustlist)
    k = clustlist(j);
    if k == 0
    else
        spikes.overcluster.colors(k,:)=jjetm(j,:);
    end
end
clusterXT(spikes, spikes.hierarchy.assigns); title('Final Clusters'); %fig6

scrn_sz = get(0,'ScreenSize');
figure('Position',[1 1+scrn_sz(4)/2 scrn_sz(3)/2 scrn_sz(4)/2])
correlations(spikes);  title('Auto- and Cross- Correlations');

figure
eps = length(unique(spikes.overcluster.assigns))/100;
h=plot([spikes.spiketimes spikes.spiketimes]', [spikes.overcluster.assigns'+eps spikes.overcluster.assigns'-eps]', '.k','markersize',1);
% h=plot(spikes.spiketimes, spikes.overcluster.assigns, 'o');
%set(h,'markerfacecolor','r','markeredgecolor','none','markersize',2);
xlabel('Time (sec)'); ylabel('Cluster #');
axis tight;

figure;
ssg_databrowse3d(spikes,spikes.hierarchy.assigns); set(gcf,'color','w'); view(-30,22);

end
