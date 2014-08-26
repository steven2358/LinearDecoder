function [cost,grad] = sparseAutoencoderLinearCost(...
    theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data)

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% calculate sparsity derivative: feedforward
m = size(data,2);
z2 = W1 * data + repmat(b1,1,m);
a2 = f(z2);
z3 = W2 * a2 + repmat(b2,1,m);
h = z3; % changed to linear

rho_hat = sum(a2,2)/m;
sparsity_deriv = beta*...
    (-sparsityParam./rho_hat + (1-sparsityParam)./(1-rho_hat));

% cost due to error
err = h - data;
cost_err = sum(sum(err.*err))/2;

% deltas
delta3 = err; % changed to linear
delta2 = (W2'*delta3 + repmat(sparsity_deriv,1,m)).*fprime(a2);

% gradients
W2grad = delta3*a2'/m + lambda*W2;
W1grad = delta2*data'/m + lambda*W1;
b2grad = mean(delta3,2);
b1grad = mean(delta2,2);

KLdiv = sparsityParam*log(sparsityParam./rho_hat) + ...
    (1 - sparsityParam)*log((1 - sparsityParam)./(1 - rho_hat));

cost_err = cost_err/m;
cost_weights = lambda/2*(sum(W1(:).^2) + sum(W2(:).^2)); % w regularization
cost_sparse = beta*sum(KLdiv); % induce "sparsity"

cost = cost_err + cost_weights + cost_sparse;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function h = f(z)
    h = sigmoid(z);
end

function fpr = fprime(a)
    fpr = a.*(1-a);
end
