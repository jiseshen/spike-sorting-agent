function [spikes] = showclust(spikes, useassigns, show);
%    SHOWCLUST  temporary utility to show clusters
%       SHOWCLUST(SPIKES, [USEASSIGNS], [SHOW]);

%   Last Modified By: sbm on Fri Sep 16 08:47:56 2005

if (nargin < 2)
	useassigns = spikes.hierarchy.assigns;
end
if (nargin < 3)
    show = unique(useassigns);
end
show = reshape(show, 1, []);

smallWindow = 0.010;

tmin = (size(spikes.waveforms,2) - spikes.threshT + 1)./spikes.Fs;
tmin = 16/spikes.Fs;
if (isfield(spikes,'options') && isfield(spikes.options, 'refractory_period'))
    tref = spikes.options.refractory_period;
else
    tref = max(2, tmin*1.5);
end
tminl = tmin - 1/spikes.Fs;

%ylims = [-225 200];
tlims = [0 max(spikes.spiketimes)];
numFigs = ceil(length(show)/5);

scrn_sz = get(0,'ScreenSize');
count = 1;
for f = 1:numFigs
    figure('Position',[1+(f-1)*(scrn_sz(3))/numFigs 1 scrn_sz(3)/numFigs scrn_sz(4)]); cmap = colormap(jet);
    curr_set = (1:5) + 5*(f-1); curr_set(curr_set>length(show))=[];
    for clust = 1:length(curr_set)
        members = find(useassigns == show(curr_set(clust)));
        memberwaves = spikes.waveforms(members,:);
        membertimes = sort(spikes.spiketimes(members));
        subplot(5,3, 3 * (clust-1) + 1);
        [n,x,y] = histxt(memberwaves);
        imagesc(x,y,n); axis xy;
        if (clust < length(show))
            set(gca,'XTickLabel',{});
        end
        %set(gca, 'YLim', ylims, 'YTickLabel', {}, 'Color', cmap(1,:));

        if (show(curr_set(clust)) ~= 0),  clustname = ['Cluster# ' num2str(show(curr_set(clust)))];
    	else,                   clustname = 'Outliers';
        end
    	hy = ylabel({clustname, ['N = ' num2str(size(members,1))]});

        subplot(5,3,3 * (clust-1) + 2);
        [a, scores(count,:)] = isiQuality(membertimes, membertimes, tmin, smallWindow, tref, spikes.Fs);
        isis = sort(diff(membertimes));   isis = isis(isis <= smallWindow);
    	isis = round(isis*spikes.Fs)/spikes.Fs;
    	smalltimes = linspace(0,smallWindow,0.33*smallWindow*spikes.Fs+1);
        isiv(count) = sum(diff(membertimes)<0.002)/length(membertimes);
    	if (~isempty(isis)), n = histc(isis,smalltimes);  else,  n = zeros(length(smalltimes));  end;
        plot(smalltimes,n);    ylim = get(gca, 'YLim');
    	patch([0 tref  tref  0]', [0 0 ylim(2) ylim(2)]', -[0.1 0.1 0.1 0.1], [0.8 0.8 0.8], 'EdgeColor', 'none');
    	patch([0 tminl tminl 0]', [0 0 ylim(2) ylim(2)]', -[0.1 0.1 0.1 0.1], [0.6 0.6 0.6], 'EdgeColor', 'none');
    	set(gca,'Xlim',[0 0.01]);
        title(['ISIv: ',num2str(isiv(count)*100),'%'])
        if (clust < length(show)),  set(gca,'XTickLabel',{});
    	else,                       xlabel('ISI (sec)');
        end

        subplot(5,3,3 * (clust-1) + 3);
        hist_size = 10; % seconds
        [n,x] = hist(membertimes,1:hist_size:max(tlims));   bar(x,n/hist_size,1.0);  shading flat; ylabel("Firing Rate (Hz)")
        set(gca,'Xlim',tlims);
        if (clust < length(show)),  set(gca,'XTickLabel',{});
    	else,                       xlabel('t (sec)');
        end
        if (clust == 1), title('Spike Times');  end;

    	% shift text to make more readable
    	set([hy], 'Units', 'char');
    	hy_pos = get(hy, 'Position');    hy_pos = hy_pos + [-6*rem(clust,2),0,0];   set(hy, 'Position', hy_pos);
        count = count + 1;
    end
    spikes.hierarchy.isiv = isiv;
    spikes.hierarchy.isis = scores(:,1);
end
