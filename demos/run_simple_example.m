% This is a script file that exemplifies the use of the Proposed-AD method.
% See the file README for more information.
% 
% Ir performs the deconvolution of a blurred version of the 'cameraman' 
% image, which was previously convolved with a 13*13 boxcar blur. The deconvolution is 
% made using the Proposed-AD method, described in [1].
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A Framework for Fast Image Deconvolution with
%        Incomplete Observations,” IEEE Trans. Image Process.,
%        to be publised.

% % % % % % % % % % % % % 
% 
% Version: 1
% 
% Can be obtained online from: 
%   https://github.com/alfaiate/DeconvolutionIncompleteObs
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2016 Miguel Simoes
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, version 3 of the License.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% 
% % % % % % % % % % % % % 
addpath('../src', '../src/utils', '../data');
% % % % % % % % % % % % % 
%
% This script has two steps. 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% I. It starts by generating the observed image. 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% The following parameters can be
% modified to change the data generation:
total_iter = 150; % Total number of iterations
BlurDim = 13; % Dimensions of the blur's support
image = 'cameraman'; % Image. Can be 'cameraman', etc. Check 'data_generation.m'
blur = 'uniform'; % Blur type. Can be 'gaussian', etc. Check 'data_generation.m'
BSNR = 50; % Blurred-Signal-to-Noise Ratio

% Generates the observed image
[ original_image, blur_kernel, observed_image ] = data_generation(image, blur, BlurDim, sqrt(BlurDim), BSNR);

% Prepare image mask
mask = ones(size(observed_image));

% Regularization parameter
lambda = 5e-6;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% II. Next, it will run pADMM.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

miu_1 = 1;
miu_2 = (BlurDim-1)/2;
[ rmse_final_ProposedAD, time_ProposedAD, ~, ~, ~, estimated_image, ~ ] = deblur_Proposed( observed_image, blur_kernel, mask, total_iter, lambda, miu_1, miu_2, original_image, 0, 0, 1, 1, 1);