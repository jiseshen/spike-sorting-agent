function clusterXT(spikes, useassigns, plotAll, show, threed);
%    clusterXT  temporary utility to show clusters
%       clusterXT(SPIKES, [USEASSIGNS], [SHOW], [THREE_D]);

%   Last Modified By: sbm on Wed Aug 10 22:15:26 2005

if (nargin < 2 || isempty(useassigns)),  useassigns = spikes.overcluster.assigns;  cmap = spikes.overcluster.colors;  end;
if (nargin < 3), plotAll = 0; end
if (nargin < 4 || isempty(show)),  show = unique(useassigns); cmap = spikes.overcluster.colors(show(show>0,:),:); end;
if (nargin < 5),  threed = 0; end

% if (nargin > 1),
%     if (isfield(spikes, 'overcluster') && all(ismember(useassigns, unique(spikes.overcluster.assigns))))
%         cmap = spikes.overcluster.colors;
%     else
%         cmap = jetm(length(show));
%     end
% end

show(show == 0) = [];

clustlist = unique(useassigns);
t = ([0:size(spikes.waveforms,2)-1]-spikes.threshT)./spikes.Fs;

%%%%%%%%%%%%%%%
cla reset; hold on;
for j = 1:length(clustlist)
    k = clustlist(j);
    members = find(useassigns == k);
    waves = spikes.waveforms(members,:);

	if (k == 0),                color = [0.2 0.2 0.2];
	elseif (ismember(k,show)),  color = cmap(j-1,:);
	else,                       color = Clgy;
	end
	
    if (~isempty(members))
        if (~threed)
            if (~plotAll)
                subsample_waves = randi(length(members),[2500 1]);
                h = mplot(t, waves(subsample_waves,:), 'Color', color);
            else
                h = mplot(t, waves(:,:), 'Color', color);
            end
            set(h, 'ButtonDownFcn', {@raise_me, h});
			if (k == 0), hout = h;
			else,        hndl(k) = h;
			end
        else
            [lh,ph] = errorarea(mean(waves,1), std(waves,1,1));
            set(lh, 'Color', brighten(color, -0.6), 'ZData', repmat(k, size(get(lh,'XData'))));
            set(ph, 'FaceColor', color, 'ZData', repmat(k, size(get(ph,'XData'))), 'FaceAlpha', 0.8);
        end
    end
end
hold off; axis tight; xlabel('Time (samples)');  ylabel('Voltage (A/D Levels)');
set(gca,'Color',[0 0 0])
if (~threed),  uistack(hndl(show), 'top');
else, cameratoolbar('SetCoordSys', 'y');
end

if ( ~threed)
    leg = cell(length(show),1);
    for k = 1:length(show),  leg{k} = num2str(sort(show(k)));  end;
	if (any(useassigns == 0))
		legend([hout, hndl(show)], cat(1, {'Outliers'}, leg));
	else
		legend(hndl(show),leg);
    end
    set(legend,'Color',[0.8 0.8 0.8])
end

ylim([-250 250])