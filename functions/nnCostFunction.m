function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

J=0;
grad=0;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


Y = y;
% Part 1 cost function for NN
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(a2 * Theta2');
size(a3);
sumM=0;

for i = 1:m
   for k = 1:num_labels
     sumM = sumM + (Y(i,k)*log(a3(i,k))+((1-Y(i,k))*log(1-(a3(i,k)))));
   endfor
endfor

[Theta1_row, Theta1_col] =  size(Theta1);
[Theta2_row, Theta2_col] =  size(Theta2);

%compute regularization for Theta1
Theta2;
regularization = 0;
for i = 1: Theta1_row
  for j = 2: Theta1_col
    regularization = regularization + Theta1(i,j)^2;
  endfor
endfor

for i = 1:Theta2_row
  for j = 2:Theta2_col
     regularization = regularization + Theta2(i,j)^2;
  endfor
endfor
%compute regularization for Theta2

regularization = (lambda/(2*m)) * regularization;

J = (sumM * (-1/m)) + regularization;


%Part 2 back propagations
% with for loop

delta_3 = 0;
delta_2 = 0;

DELTA_2 = 0;
DELTA_3 = 0;
for t = 1:m
  a1_t = a1(t,:);
  z2_t = z2(t,:);
  a2_t = a2(t,:);
  z3_t = z3(t,:);
  a3_t = a3(t,:);
  y_t = Y(t,:);
  delta_3 = a3_t - y_t;
  delta_3 = delta_3';
  delta_2 = (Theta2' * delta_3)(2:end,:) .* sigmoidGradient(z2_t');
  
  grad_1 = delta_2 * a1_t;
  grad_2 = delta_3 * a2_t;
  DELTA_2 = DELTA_2 + grad_1;
  DELTA_3 = DELTA_3 + grad_2;
endfor


DELTA_2 = (1/m)*DELTA_2;
DELTA_3 = (1/m)*DELTA_3;

DELTA_2_regularization_sum = (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
DELTA_2 = DELTA_2 + DELTA_2_regularization_sum;

DELTA_3_regularization_sum = (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];
DELTA_3 = DELTA_3 + DELTA_3_regularization_sum;

%add regularization

Theta1_grad = DELTA_2;
Theta2_grad = DELTA_3;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end