# Neural-Network-to-predict-handwritten-digits

## Description

This project implements a neural network for pattern recognition using MATLAB. The neural network is trained to classify 1x400 images of digits into their respective labels. The project includes the training of the network, evaluation of its performance, and the ability to customize the network with regularization.

## Motivation

The motivation behind this project is to explore and understand the concept of neural networks and their applications in pattern recognition. By building and training a neural network, one can gain insights into how these models can learn and classify patterns based on the provided data.

## Problem Solving

The neural network developed in this project solves the problem of digit recognition. Given an image of a digit, the network is trained to accurately predict the corresponding label for that digit. This can have various practical applications, such as optical character recognition (OCR) systems and automated data entry.

## Learning Outcomes

Through this project, I have learned several key concepts and techniques related to neural networks and pattern recognition. Specifically, I have gained knowledge in the following areas:

1. Neural network architecture: I have learned how to design and structure a neural network model for pattern recognition tasks. This includes defining the number of hidden neurons and selecting appropriate transfer functions.

2. Regularization: I have explored the concept of regularization in neural networks and its role in controlling overfitting. I have learned how to incorporate regularization into the network using the MATLAB neural network toolbox.

3. Training and evaluation: I have gained hands-on experience in training neural networks using the provided data. I have learned how to evaluate the network's performance and measure its accuracy using appropriate metrics.

4. MATLAB neural network toolbox: I have familiarized ourselves with the functionalities and features of the MATLAB neural network toolbox. This includes using the toolbox to create, train, and save neural network models.

By working on this project, I have developed a solid foundation in neural networks and pattern recognition, which can be further applied to more complex tasks and datasets.

## Code Explanation

```matlab
clear;
load('data1.mat');
Y = (1:10) == y;
```

The code loads the data from the 'data1.mat' file and converts the labels into a binary matrix representation for training the neural network.

```matlab
i = randi(length(y));
ysim = net(X(i,:)');
[~,class] = max(ysim);
imshow(reshape(X(i,:),20,20))
fprintf('True class: %d  |  Predicted class: %d | Probability of match: %.1f%%',y(i),class,100*ysim(class));
```

These lines select a random example from the dataset, feed it to the trained neural network, and display the predicted class along with the true class and the probability of match.

```matlab
lambda = 0.01;
MaxIter = 90;
net = imageNet(X,Y,lambda,MaxIter);
[~,ysim] = max(net(X'));
fprintf('Training accuracy: %g%%',100*sum(y == ysim')/length(y));
```

This part of the code trains a custom neural network with regularization. It sets the lambda (regularization parameter) and MaxIter (maximum number of iterations) values, and then calls the 'imageNet' function to build and train the neural network. The training accuracy is then calculated and displayed.

```matlab
function net = imageNet(X,Y,lambda,MaxIter) 
% Build, train, and return a neural network model to classify the 1x400 images
% of digits in X with labels in Y

% ... (function body omitted for brevity) ...
end
```

The 'imageNet' function is a helper function that constructs and trains a neural network model. It takes the input data X, the corresponding labels Y, the regularization parameter lambda, and the maximum number of iterations MaxIter as inputs. The function builds and trains the neural network using the provided parameters and returns the trained network model.


## Conclusion

In conclusion, this project has provided an opportunity to delve into the world of neural networks for pattern recognition. By building and training a neural network model, we have gained insights into the inner workings of these models and their applications in solving classification problems.
