function plot_rsquare(t,ressq)
% function plot_rsquare(t,ressq)
% plots r^2 color map 
% t - time samples
% ressq - r2 value

data2plot=ressq';
data2plot=cat(2, data2plot, zeros(size(data2plot, 1), 1));
data2plot=cat(1, data2plot, zeros(1, size(data2plot, 2)));
xData=t;

%size(xData)
xData(end+1) = xData(end) + diff(xData(end-1:end));

Nch=size(ressq,2);

surf(xData, [1:Nch + 1], data2plot,'EdgeColor','none'); 
axis tight;
view(2);
colormap jet;
colorbar;

% may define max and min color bar
% caxis([0 0.1])

