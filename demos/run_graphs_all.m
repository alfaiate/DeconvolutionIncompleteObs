% This is a script file that exemplifies the use of the Proposed-AD method.
% See the file README for more information.
% 
% It runs four scripts. Each one corresponds to the 'running time' graphs 
% given in [1], i.e., the RMSE of the estimated images as a function of
% running time, for the various tested methods.
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

total_iter = 1e6; % Total number of iterations
error_min = 1e-3; % Minimum RMSE to terminate the algorithms

fprintf('Experiment: Deconvolution with unknown boundaries\n')
run_graph_deblurring

fprintf('Experiment: Deconvolution with inpainting\n')
run_graph_inpainting

fprintf('Experiment: Superresolution\n')
run_graph_sr

fprintf('Experiment: Demosaicing\n')
run_graph_demosaicing