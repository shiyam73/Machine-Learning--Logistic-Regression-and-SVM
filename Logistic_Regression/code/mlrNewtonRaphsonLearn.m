function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


train_data_with_bias = [ones(size(X,1),1) X];

w_old = initial_W;
I = eye(10);
size_row = size(train_data_with_bias,1);
size_col = size(train_data_with_bias,2);

    
for k = 1 : n_iter
    W = reshape(w_old, size(X, 2) + 1, size(T, 2));
    actfunc_matrix = exp(train_data_with_bias * W);
    actfunc_summation = sum(actfunc_matrix,2);
    actfunc_sum_repmatrix = repmat(actfunc_summation,1,size(actfunc_matrix,2));

    y = actfunc_matrix./actfunc_sum_repmatrix;
        %Computing the Hessian
    for i=1:10
        for j=1:10     
            H_oneblock = zeros(size_col,size_col);
            for n=1:size_row
                ynk_term = y(n,i)*(I(i,j)-y(n,j));
                x_transpose_into_x = (train_data_with_bias(n,:))'*(train_data_with_bias(n,:));
                H_oneblock = H_oneblock + ynk_term*x_transpose_into_x; 
            end
            if(j==1)
                H_tenblocks = H_oneblock;
            else
                H_tenblocks_tmp = H_tenblocks;
                H_tenblocks = [H_tenblocks_tmp,H_oneblock];
            end
        end
        if(i==1)
            H_hundredblocks = H_tenblocks;
        else
            H_hundredblocks_tmp = H_hundredblocks;
            H_hundredblocks = [H_hundredblocks_tmp;H_tenblocks];
        end
    end

    H = H_hundredblocks;


    %Computing the Gradiance
    grad_matrix = ((y - T)'*train_data_with_bias)';

    % Unroll gradients to single column vector
    error_grad = grad_matrix(:);
    
    w_old = w_old - mldivide(H , error_grad) ;
end

W = reshape(w_old, size(X, 2) + 1, size(T, 2));


end

