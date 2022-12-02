% Example that shows how to apply the statistical spatial filter FBC (Fisher Criterion Beamformer) 
% proposed in [1] (section 3.2.2): 
% 
% [1] Gabriel Pires, Urbano Nunes and  Miguel Castelo-Branco (2011), "Statistical Spatial Filtering for 
%     a P300-based BCI: Tests in able-bodied, and Patients with Cerebral Palsy and Amyotrophic Lateral 
%     Sclerosis", Journal of Neuroscience Methods, Elsevier, 2011, 195(2), 
%     Feb. 2011: doi:10.1016/j.jneumeth.2010.11.016
%     https://www.sciencedirect.com/science/article/pii/S0165027010006503?via%3Dihub

% The example uses 2 different datasets that can be used for testing: 
% - dataset 1 : P300 oddball elicited in a communication speller (LSC speller)
% - dataset 2 : Error-related potentials (ErrP) elicited to detect and correct errors in LSC speller
% For more information about LSC speller and datasets please check: 
% [2] https://ieee-dataport.org/open-access/error-related-potentials-primary-and-secondary-errp-and-p300-event-related-potentials-%E2%80%93
% [3] https://ieeexplore.ieee.org/abstract/document/8048036
%
% For the purpose of this example, it is useful to know that the datasets use:   
% - 12 EEG channels: order [1-Fz 2-Cz 3-C3 4-C4 5-CPz 6-Pz 7-P3 8-P4 9-PO7 10-PO8 11-POz 12-Oz]
% - sampling frequency: fs = 256 Gz
% - dataset 1 is filtered [0.5-30] Hz
% - dataset 2 is filtered [1-10] Hz
% - epochs are all 1 second long
%
% - the FCB toolbox is in folder 'FCB_toolbox' 
% - some analysis files are in folder 'functions'
% - this example is self-explanatory from code and comments - just run and select the dataset
% - to show the discrimination effect of FCB r-square is applied on data before and after FCB filtering 
%
% If you use the FCB toolbox please refer to [1]
% If you use these datasets please refer to [2] and [3]

% Gabriel Pires, April 2022

addpath('rsquare', 'FCB_toolbox')
clc

%DATASET SELECTON 
% dataset_sel = 1 (P300 dataset) / dataset_sel = 2 (ErrP dataset)
dataset_sel = input('SELECT THE DATASET TO TEST (1: P300 dataset /  2 : ErrP dataset:  ');

if dataset_sel == 1 
    data=load('data\P300_LSC_dataset.mat');
    yt = data.P300_LSC_dataset.ytarget;        %target samples (1 second  epochs)
    ynt = data.P300_LSC_dataset.yNONtarget;    %Non-target samples (1 second  epochs)
    %size(yt)     % 12 channels x 256 time samples x 90   target trials
    %size(ynt)    % 12 channels x 256 time samples x 2430 NONtarget trials
    clear data
    
    %the dataset is very imbalanced, so we may limit NONtarget trials to fewer trials , let's say 840
    ynt = ynt(:,:, 1:840);
    
    fprintf('P300 DATASET\n')
    fprintf('Variable yt:   P300 epochs     - %d channels x %d time samples x %d target trials \n', size(yt,1),size(yt,2),size(yt,3));
    fprintf('Variable ynt:  standard epochs - %d channels x %d time samples x %d NONtarget trials \n', size(ynt,1),size(ynt,2),size(ynt,3));
end

if dataset_sel == 2 
    data=load('data\ErrP_LSC_dataset.mat');
    yt = data.ErrP_LSC_dataset.ErrP;        %ErrP samples (1 second  epochs)
    ynt = data.ErrP_LSC_dataset.CorrERP;    %Correct Event Related Potentials samples (1 second  epochs)
    %size(yt)     % 12 channels x 256 time samples x 28   Error trials 
    %size(ynt)    % 12 channels x 256 time samples x 168  Correct trials
    clear data
    
    fprintf('ERRP DATASET\n')
    fprintf('Variable yt:   ErrP epochs         - %d channels x %d time samples x %d Error trials \n', size(yt,1),size(yt,2),size(yt,3));
    fprintf('Variable ynt:  Correct  ERP epochs - %d channels x %d time samples x %d Correct trials \n', size(ynt,1),size(ynt,2),size(ynt,3));
end

%% Average of data (targets and non-targets)
yt_mean = mean(yt,3);
yt_std  = std(yt,0,3);
ynt_mean = mean(ynt,3);
ynt_std  = std(ynt,0,3);

fs = 256;                   % sampling frequency: 256 Hz
t=(0:256-1)*1/fs;           % 1 second trial

hFig=figure;
set(hFig,'Position',[0 0 600 1200]);
subplot(3,1,1)
plot(t,yt_mean(1,:),'b','linewidth',3)
hold on
plot(t,yt_mean(2,:),'b-');plot(t,yt_mean(3,:),'b-');plot(t,yt_mean(4,:),'b-*');plot(t,yt_mean(5,:),'g','linewidth',3);plot(t,yt_mean(6,:),'r-');plot(t,yt_mean(7,:),'r-.'); plot(t,yt_mean(8,:),'r-*'); plot(t,yt_mean(9,:),'r.-'); plot(t,yt_mean(10,:),'m','linewidth',3); plot(t,yt_mean(11,:),'m-','linewidth',3); plot(t,yt_mean(12,:),'m-.','linewidth',3)
xlabel('time (s)'); ylabel('amplitude (\muV)')
title('Average - targets')
legend('Fz','Cz', 'C3', 'C4', 'CPz', 'Pz', 'P3', 'P4', 'PO7', 'PO8', 'POz', 'Oz')
hold off

subplot(3,1,3)
if dataset_sel == 1 
    plot(t,yt_mean(2,:),'r','linewidth',3); hold on; plot(t,ynt_mean(2,:),'b','linewidth',3)
    legend('Target','Standard')
    xlabel('time (s)'); ylabel('amplitude (\muV)')
    title('Average - Channel Cz')
end
if dataset_sel == 2 
    plot(t,yt_mean(1,:),'r','linewidth',3); hold on; plot(t,ynt_mean(1,:),'b','linewidth',3)
    legend('Error','Correct')
    xlabel('time (s)'); ylabel('amplitude (\muV)')
    title('Average - Channel Fz')
end


%% analysis of feature discrimination with r-square of raw data
N_ch = size(yt, 1);     %#channels
N_samp = size(yt, 2);   %#time sample
rsq = zeros(N_samp, N_ch);   %initialize variable
for ch=1:N_ch         
    for samp=1:N_samp   
       rsq(samp, ch)=rsquare(yt(ch, samp,:),ynt(ch, samp,:));
    end
end
subplot(3,1,2)
plot_rsquare(t,rsq); xlabel('time (s)'); ylabel('channels');
title('Statistical r^2 between target and non-target');   
clear rsq

%% ----------------------------------------------------------- 
% obtaining FCB statistical spatial filters
[U V] = FCB_spatial_filters(yt, ynt, 0.1); 

% Projections obtained from spatial filters
[ytf yntf] = FCB_projections(yt, ynt,U); 

% for classication we can use one projection or two concatenated
% projections, as they are the most discriminative as seen below
% -------------------------------------------------------------

%% Analyzing the effect of FCB spatial filtering
ytf_mean = mean(ytf,3);
ytf_std  = std(ytf,0,3);
yntf_mean = mean(yntf,3);
yntf_std  = std(yntf,0,3);

hFig=figure;
set(hFig,'Position',[600 0 600 1200]);
subplot(3,1,1)
plot(t,ytf_mean(1,:),'r','linewidth',3)
hold on
plot(t,ytf_mean(2,:),'b-','linewidth',3);plot(t,ytf_mean(3,:),'k');plot(t,ytf_mean(4,:),'k');plot(t,ytf_mean(5,:),'k');plot(t,ytf_mean(6,:),'k');plot(t,ytf_mean(7,:),'k'); plot(t,ytf_mean(8,:),'k'); plot(t,ytf_mean(9,:),'k'); plot(t,ytf_mean(10,:),'k'); plot(t,ytf_mean(11,:),'k'); plot(t,ytf_mean(12,:),'k')
xlabel('time (s)'); ylabel('amplitude (\muV)')
title('Average - FCB projected targets')
legend('Proj 1','Proj 2','other proj'); 
hold off

subplot(3,1,3)
if dataset_sel == 1 
    plot(t,ytf_mean(1,:),'r','linewidth',3); hold on; plot(t,yntf_mean(1,:),'b','linewidth',3)
    legend('Target','Standard')
    xlabel('time (s)'); ylabel('amplitude (\muV)')
    title('Average - 1st Projection')
end
if dataset_sel == 2 
    plot(t,ytf_mean(1,:),'r','linewidth',3); hold on; plot(t,yntf_mean(1,:),'b','linewidth',3)
    legend('Error','Correct')
    xlabel('time (s)'); ylabel('amplitude (\muV)')
    title('Average - 1st projection')
end


%% analysis of feature discrimination with r-square of spatially filtered data
N_ch = size(ytf, 1);     %#channels
N_samp = size(ytf, 2);   %#time sample
rsq = zeros(N_samp, N_ch);   %initialize variable
for ch=1:N_ch         
    for samp=1:N_samp   
       rsq(samp, ch)=rsquare(ytf(ch, samp,:),yntf(ch, samp,:));
    end
end
subplot(3,1,2)
plot_rsquare(t,rsq); xlabel('time (s)'); ylabel('projections');
title('Statistical r^2 between FCB projected targets and non-targets');   
clear rsq