function g = sigmoid(z)
% sigmoid computes sigmoid functoon
% Notice that z can be a scalar, a vector or a matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = 1.0 ./ (1.0 + exp(-z));
end
