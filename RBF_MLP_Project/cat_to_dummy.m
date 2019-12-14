function [ret_data] = cat_to_dummy(data,row)
%CAT_TO_DUMMY Converts a row from categorical data to dummy variable
% Note: taking the easy route and not worrying about row == 1 case. 
%       This case won't be hit with the heart dataset

prefix = data( 1 : row - 1, :);
data_size = size(data);

if row < data_size(1)
    suffix = data(row + 1 : data_size(1), :);  
end

data_row = data(row,:);

dummies = dummyvar(data_row)';

ret_data = [prefix;dummies];

if row < data_size(1)
   ret_data = [ret_data; suffix]; 
end

end

