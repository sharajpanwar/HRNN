% Creating training and test data for 3 fold cross validation
% 'Alerts samples =  3x250', 'Drowsy sample = 3x826'
clear Inputs K1 K2 K3 Labels Alerts_change_shape Drowsy_change_shape
x = [1: 752]; x = x(randperm(length(x)));
target1 = Alerts_hrnn(x(1:250), :, :);
target2 = Alerts_hrnn(x(251:500), :, :);
target3 = Alerts_hrnn(x(501:750), :, :);

clear x; % 3-fold non_target
x = [1: 2479]; x = x(randperm(length(x)));
non_target1 = Drowsy_hrnn(x(1:826), :, :);
non_target2 = Drowsy_hrnn(x(827:1652), :, :);
non_target3 = Drowsy_hrnn(x(1653:2478), :, :);
   
%creating training data => tr_drive_hrnn_1, tr_drive_hrnn_2, tr_drive_hrnn_3
tr_drive_hrnn_1 =  [target2;target3;non_target2;non_target3];
tr_drive_hrnn_2 =  [target1;target3;non_target1;non_target3];
tr_drive_hrnn_3 =  [target1;target2;non_target1;non_target2];

%creating test data =>  te_drive_hrnn_1, te_drive_hrnn_2, te_drive_hrnn_3
te_drive_hrnn_1 = [target1;non_target1];
te_drive_hrnn_2 = [target2;non_target2];
te_drive_hrnn_3 = [target3;non_target3];

clear Alerts_hrnn Drowsy_hrnn target1 target2 target3 non_target1 non_target2 non_target3 x
% each training data file (tr) has first 500 samples target and rest 1652 non-target
% each test data file (te) has first 250 samples target and rest 826 non-target

%saving final data 
save tr_drive_hrnn_1.mat tr_drive_hrnn_1 -v7.3
save tr_drive_hrnn_2.mat tr_drive_hrnn_2 -v7.3
save tr_drive_hrnn_3.mat tr_drive_hrnn_3 -v7.3
save te_drive_hrnn_1.mat te_drive_hrnn_1 -v7.3
save te_drive_hrnn_2.mat te_drive_hrnn_2 -v7.3
save te_drive_hrnn_3.mat te_drive_hrnn_3 -v7.3
