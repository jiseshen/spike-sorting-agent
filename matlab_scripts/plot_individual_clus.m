function plot_individual_clus(spikes, useassigns, show)

if (nargin < 2 || isempty(useassigns)),  useassigns = spikes.overcluster.assigns;  cmap = spikes.overcluster.colors;  end;
if (nargin < 3 || isempty(show)),  show = unique(useassigns); show(show == 0) = []; cmap = spikes.overcluster.colors(show,:);end;

% if (nargin > 1),
%     if (isfield(spikes, 'overcluster') && all(ismember(useassigns, unique(spikes.overcluster.assigns))))
%         cmap = spikes.overcluster.colors;
%     else
%         cmap = jetm(length(show));
%     end
% end

% added by shreesh on 2010.06.07 to accomodate cluster renaming
% jjetm = colormap(jet);
% L=length(jjetm);
% cmap  = [cmap;jjetm(randperm(L),:)];

clustlist = unique(useassigns);
t = ([0:size(spikes.waveforms,2)-1]-spikes.threshT)./spikes.Fs;

%%
n_clus=length(clustlist);
nrows=ceil(n_clus/4); ncols=4;
for j = 1:length(clustlist) % unique cluster
    k = clustlist(j);
    members = find(useassigns == k); % getting waveforms idxs belong to cluster k
    waves = spikes.waveforms(members,:); % getting waveforms belong to cluster k

   % coloring clusters
	if (k == 0),                color = [0 0 0];
	elseif (ismember(k,show)),  color = cmap(j-1,:); % plotting all but only coloring in "show" waves
	else,                       color = Clgy;
	end
	
    if (~isempty(members))
        subplot(nrows, ncols, j);
        subsample_waves = randi(length(members),[2500 1]);
        h = mplot(t, waves(subsample_waves,:), 'Color', color); % plots all waveforms of cluster k
        title(['Cluster #',int2str(k),' (n=',int2str(length(members)),')']);
        set(h, 'ButtonDownFcn', {@raise_me, h});
		if (k == 0), hout = h;
		else, hndl(k) = h;
		end
    else
        [lh,ph] = errorarea(mean(waves,1), std(waves,1,1));
        set(lh, 'Color', brighten(color, -0.6), 'ZData', repmat(k, size(get(lh,'XData'))));
        set(ph, 'FaceColor', color, 'ZData', repmat(k, size(get(ph,'XData'))), 'FaceAlpha', 0.8);
    end
end  
end

