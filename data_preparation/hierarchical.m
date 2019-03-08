
% HRNN data: %functuion fo Reshaping EEG epoch obtained from epoched_data 
% (Alerts_change_shape, Drowsy_change_shape) from 64x640 to 10x4096

function hierarchical = hrnn(data, c)

for i = 1:c;
         sample = squeeze(data(i, :, :));
         k1 = 1; chunks=[]; 
         for j = 1:10;
             k =64*j;
             chunk_sample = sample(:,k1:k);
             chunk_sample = chunk_sample';
             chunk{1,j} = reshape(chunk_sample', 1, 4096);
             k1=k1+64;
             chunks = cat(1, chunk{:});
         end
         hierarchical(i,:,:) = chunks;
end
end
%  the function can be used as 
% Alerts_hrnn = hierarchical(Alerts_change_shape, K3);
% Drowsy_hrnn = hierarchical(Drowsy_change_shape, K2);
% K2 and K3 are from epoched_data.m