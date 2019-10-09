 
clear all;
close all;
 
%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:
 
  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.
 
 
%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.
 
% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);
 
% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);
 
% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);
 
data = train;

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images
 
%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image
 
figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.
 
%% After importing, the array train consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.
 
%% This next section of code calls the three functions you are asked to specify
 
k= 10; % set k
max_iter= 10; % set the number of iterations of the algorithm
 
%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.
 
centroids=initialize_centroids(train,k);
 
%% Initialize an array that will store k-means cost at each iteration
 
cost_iteration = zeros(max_iter, 1);
 
%% This for-loop enacts the k-means algorithm
 
% FILL THIS IN
%       initialize
initialize_centroids(data,k);
for i = 1:max_iter
%       assign to centroids / vector
assign_vector_to_centroid(data,centroids,k);
%        update centroids 
update_Centroids(data,k);
end

%% This section of code plots the k-means cost as a function of the number
% of iterations
 
% FILL THIS IN 

% cost_iteration is the variable
figure;

 
%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.
 
figure;
colormap('gray');
 
plotsize = ceil(sqrt(k));
 
for ind=1:k
    
    centroid=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centroid,[28 28])');
    title(strcat('Centroid ',num2str(ind)))
 
end
 
%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% *Feel free to experiment.*
% Note that this function takes two inputs and emits one output (y).

function y = initialize_centroids(train,k)
 
meanVector = mean(train); % creates the mean vector from the training set
 
distance = zeros(1500,1); % creates a 1500x1 arrays of zeros called distance
 
for i = 1:1500 % for loop!
    
    vectorI = train(i,785); % isolates vector i from the training set
    
    distance(i,1) = norm(vectorI - meanVector); % calculates the distance from vector i to the mean vector and then inputs that value into the distance array at i
    
end
 
trainWithDistance = [train,distance]; % creates array trainWithDistance that has train and distance side-by-side
 
trainWithDistanceSorted = sortrows(trainWithDistance,786); % sorts trainWithDistance based on distance values
 
centroids = trainWithDistanceSorted(end-(k-1):end,1:784); % creates array centroids that is composed of the k vectors that are the farthest from the mean of all vectors
 
centroidKValuesTranspose = 1:k; % creates column vector centroidKValuesTranspose with entries i = 1, 2, ..., k-1, k
 
centroidKValues = centroidKValuesTranspose';
 
centroidsWithLabel = [centroids,centroidKValues]; % concatenates centroids and centroidKValues side-by-side to create centroidWithLabel
 
y = centroidsWithLabel; % assigns centroidsWithLabel to y, the output of this function
 
end


%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.
 
function [index, vec_distance] = assign_vector_to_centroid(data,centroids,k)
    smallestdist = norm(data - centroids(1,:));
    for i = 1:k
        dist = norm(data - centroids(i,:));
        if (dist < smallestdist)
            smallestdist = dist;
            index = i;
        end
    end
    vec_distance = smallestdist;
end
 
 
%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.
 
function new_centroids=update_Centroids(data,k)
    new_centroids = zeros(k,784);
    for i = 1:k
        % trying to fix the error i'm getting the "785==i"
        %for j = 1:100 %100 is a placeholder, need size of data?
        %    if (data(j,785) == i)
                
        %    end
        %end
        vectorsWithCentroidI = data(:,785==i);
        new_centroids(i,:) = mean(vectorsWithCentroidI(:,:));
    end
end
