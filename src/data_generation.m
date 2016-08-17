function [ original_image, blur_kernel, observed_image ] = data_generation( image, blur_type, BlurDim, param, BSNR )
%data_generation - Generates a blurred image
% 
% function [ original_image, blur_kernel, observed_image ]
%           = data_generation( image, blur_type, BlurDim, param, BSNR )
% 
% Input: 
% image: possible options are 'lena', 'camera', 'pavia', 'parrot', 'mandril_color'
% blur_type: 'uniform', 'focus', 'gaussian', 'motion', 'positive_def'
% BlurDim:  blur's support size,
% param: parameter for the fspecial funtion,
% BSNR:  Blurred Signal-to-Noise Ratio.
% 
% 
% Output: 
% original_image: Sharp image, 
% blur_kernel: filter, 
% observed_image: blurred image+noise.

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
% 
% Load image
switch image
    case 'lena'
        x0 = double(imread('lena256.bmp'));
        x0 = x0-min(x0(:));
        x0 = x0/max(x0(:));   % x0 in the range [0,1]
        
    case 'cameraman'
        x0 = double(imread('cameraman.tif'));
        x0 = x0-min(x0(:));
        x0 = x0/max(x0(:));   % x0 in the range [0,1]
        
    case 'pavia'
        x0 = double(imread('pavia.bmp'));
        x0 = x0-min(x0(:));
        x0 = x0/max(x0(:));   % x0 in the range [0,1]

    case 'parrot'
        x0 = double(imread('parrot.tif'));
        x0 = x0-min(x0(:));
        x0 = x0/max(x0(:));   % x0 in the range [0,1]
        
    case 'mandril_color'
        x0 = double(imread('mandril_color.tif'));
        x0 = x0-min(x0(:));
        x0 = x0/max(x0(:));   % x0 in the range [0,1]
end

% Blur image
switch blur_type
    case 'focus'
        blur_kernel = fspecial('disk', BlurDim/2);
        
    case 'gaussian'
        blur_kernel = fspecial('gaussian', BlurDim, param);
        
    case 'motion'
        blur_kernel = fspecial('motion', BlurDim, param);
        
    case 'uniform'
        blur_kernel = fspecial('average', BlurDim);
        
    case 'positive_def'
        blur_aux = fspecial('gaussian', BlurDim/2, sqrt(BlurDim/2));
        blur_aux2 = convn(blur_aux, blur_aux, 'full');
        delta = zeros(size(blur_aux2));
        delta(1, 1) = param;
        blur_kernel = blur_aux2 + fftshift(delta);
        blur_kernel = blur_kernel./sum(blur_kernel(:));

end
y = convn(x0, blur_kernel, 'valid');

% Crop the original image to the same size as the blurred one
[sz1_y, sz2_y, ~] = size(y);
[sz1_x0, sz2_x0, ~] = size(x0);
original_image = imcrop(x0, [(sz2_x0-sz2_y)/2+1 (sz1_x0-sz1_y)/2+1 sz2_y-1 sz1_y-1]);

% Add noise
sigma = sqrt(var(y(:)) / 10^(BSNR/10));
rng('default');
observed_image = y + sigma * randn(size(y));