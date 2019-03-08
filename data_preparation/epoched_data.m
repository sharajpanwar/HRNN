%epoching of EEG data (sampling frequency 128) to 5 second epoch 
%loading dataset
load('/home-new/ijz121/codes_matlab/Experiment_XB_Baseline_Driving.mat');
% The sampling frequency is 128, Input format is (channel,time-steps, samples) 
%taking off 5 second epoch%taking off 5 second epoch
for i = 1:108;
     Inputs{i} = Inputs{i}(:,14:653,:);                                                                                                                                                                                                                                    
end
% if the response time <0.75 its an alert
% if the response time >2.1 its a drowsy
% if the response time is inbetween 0.7 and 2.1 is neither
%converting labels to alert(3), drowsy(2) and mid-values(1) 
class_labels=cell(1, 108);
 for i = 1 : 108;
class_labels{i}(Labels{i}<2.1 & Labels{i}>0.75)=1; 
class_labels{i}(Labels{i}>2.1)=2;
class_labels{i}(Labels{i}<0.75)=3;
 end 
 %checking number of sampels K1=mid values, K2=Drowsy, K3=Alerts
K1 = 0; K2 = 0;K3 = 0;
for i = 1 : 108;                                                                                                           
N1=numel(class_labels{i}(class_labels{i}==1));
N2=numel(class_labels{i}(class_labels{i}==2));
N3=numel(class_labels{i}(class_labels{i}==3)) ;
K1=K1+N1; K2=K2+N2; K3=K3+N3;
end
%getting rid of mid value data and keeping alert and drowsy data only
for i = 1:108;
    id = find(class_labels{i}==1);
    Inputs{i}(:, :, id) = [];
    class_labels{i}(:,id) = [];    
end
% We have a biased data with less Alert samples compared to Drowsy samples, 
% The ratio is alert:drowsy::1:3.29
% This is the practical nature of the data which we want to preserve so we will keep it like that
c=size(class_labels);
% seperating alerts and drowsy
Alerts = []; Drowsy= [];
for i = 1:c(2);
    id = find(class_labels{i}==3);
    id2 = find(class_labels{i}==2);
    alert = Inputs{i}(:, :, id);
    drowsy = Inputs{i}(:, :, id2);
    Alerts = cat(3, Alerts, alert);
    Drowsy = cat(3, Drowsy, drowsy);   
    end   
%switching axis/dimension to (epochs, channels, time-samples)

Alerts_change_shape= permute(Alerts, [3, 1, 2]);
Drowsy_change_shape= permute(Drowsy, [3, 1, 2]);
clear N1 N2 N3 alert Alerts c class_labels Drowsy  i  drowsy id id2 
% HRNN data: To obtained hrnn data we will change the EEG epoch shape from 64x640 to 10x4096
% we will use function defined in 'hierarchical.m '
