function [ rmse, elapsed, snr, isnr, ssim, estimated_image, estimated_image_full ] = deblur_Proposed( observed_image, blur_kernel, mask, maxiter, lambda, miu_1, maxBGS_pass, original_image, final_image, min_error, show_image, print_status, compute_quality )
%deblur_Proposed - Corresponds to the Proposed-AD method discussed in [1]
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A Framework for Fast Image Deconvolution with
%        Incomplete Observations,” IEEE Trans. Image Process.,
%        to be published.
% 
% function [ rmse, elapsed, snr, isnr, ssim, estimated_image, estimated_image_full ] 
%           = deblur_Proposed( observed_image, blur_kernel, mask, maxiter, lambda, miu_1, 
%               maxBGS_pass, original_image, final_image, min_error, show_image, 
%               print_status, compute_quality )
% 
% Input: 
% observed_image: blurred image
% blur_kernel: filter
% mask: mask with unobserved pixels
% maxiter: maximum number of iterations
% lambda: regularization parameter
% miu_1: penalization parameter
% maxBGS_pass: maximum number of BGS passes
% original_image: used to compute SNR, ISNR, SSIM
% final_image: used to compute the RMSE (corresponds to the solution/reference image)
% min_error: stop the algorithm when the RMSE is below this threshold
% show_image: flag to show the image every 'show_image' iterations
% print_status: flag to print the values of the different quality metrics every 'print_status' iterations
% compute_quality: flag to compute the different quality metrics every 'compute_quality' iterations
% 
% Output: 
% rmse
% elapsed: running time
% snr
% isnr
% ssim
% estimated_image: cropped image (corresponding to the dimensions of the original image)
% estimated_image_full: image with estimated boundaries
 
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

% This script has two steps. 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% I. It starts by initializing the variables used by pADMM.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

if (show_image)
    scrsz = get(0,'ScreenSize');
    sfigure(1); set(1,'OuterPosition',[round(scrsz(3)/3) round(scrsz(4)*0.03)+round((scrsz(4)/2)*(1-0.05/2)) round(scrsz(3)/3) round((scrsz(4)/2)*(1-0.05/2))]); % [left bottom width height]
    if (compute_quality)
        sfigure(2); set(2,'OuterPosition',[round(scrsz(3)*2/3) round(scrsz(4)*0.03)+round((scrsz(4)/2)*(1-0.05/2)) round(scrsz(3)/3) round((scrsz(4)/2)*(1-0.05/2))]);
        sfigure(3); set(3,'OuterPosition',[round(scrsz(3)*2/3) round(scrsz(4)*0.03) round(scrsz(3)/3) round((scrsz(4)/2)*(1-0.05/2))]);
    end
    if numel(final_image) > 1
        sfigure(4); set(4,'OuterPosition',[round(scrsz(3)/3) round(scrsz(4)*0.03) round(scrsz(3)/3) round((scrsz(4)/2)*(1-0.05/2))]); % [left bottom width height]
    end
end

% Half of the blur's support.
hsize_h = floor(size(blur_kernel, 2)/2);
hsize_w = floor(size(blur_kernel, 1)/2);

% Pad image
padval = 'replicate';
y = padimage(observed_image, [hsize_h hsize_w], padval);

% Prepare image mask
imagemask = padimage(mask, [hsize_h hsize_w], 0);
padmask = 1 - imagemask;
Mty = y .* imagemask;

% Properly sized blur filter
id = zeros(size(y));
id(1, 1) = 1;
h = imfilter(id, blur_kernel, 'circular', 'conv');
h = h / sum(h(:)); % Filter calibration

% Define the difference operators as filters
g = zeros([size(y) 2]);
g(1,1) = -1;                        % Vertical
g(end,1) = 1;
gf = fft2(g);
g(1,1,2) = -1;                     % Horizontal
g(1,end,2) = 1;
gf(:,:,2) = fft2(g(:,:,2));
gfc = conj(gf);
gf2 = gfc .* gf;
gf2sum = sum(gf2, 3);

% Initialization
hf = fft2(h);
hfc = conj(hf);           % Conjugate
hf2 = hfc .* hf;          % Square of absolute value

snr = zeros(1, maxiter);
isnr = zeros(1, maxiter);
ssim = zeros(1, maxiter);
elapsed = zeros(1, maxiter);
rmse = zeros(1, maxiter);

iter = 1;
alpha = 2; % Overrelaxation factor

% Warm up tic/toc.
tic();
elapsed(1) = toc();
tic();
elapsed(1) = toc();

y_estimate_borders = padmask .* y;
tau = 1e-3;

% Initialization of z
tic();
for i = 1:100

    y = Mty + padmask .* y_estimate_borders;
    yf = fft2(y);
    xf = (1./(hf2 + tau) + tau).*(hfc.*yf);
    y_estimate_borders = real(ifft2(xf .* hf));
    
end

pre_elapsed = toc();

yf = fft2(y);
x = real(ifft2(hfc.*xf));

v(:,:,1) = [diff(x,1,1); x(1,:) - x(end,:)];
v(:,:,2) = [diff(x,1,2), x(:,1) - x(:,end)];
d = v;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% II. Partial ADMM.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

while iter <= maxiter
    tic();
        
    csi(:,:,1) = v(:,:,1) + d(:,:,1);   % vertical
    csi(:,:,2) = v(:,:,2) + d(:,:,2);   % horizontal
    d_old = d;
    
    % Primal
    % BGS cycle
    for GS_pass = 1:maxBGS_pass
        
        y_estimate_borders_old = y_estimate_borders;
                 
        rf = miu_1 * fft2([csi(end,:,1) - csi(1,:,1); -diff(csi(:,:,1),1,1)] ...
            + [csi(:,end,2) - csi(:,1,2), -diff(csi(:,:,2),1,2)]) + hfc .* yf;
        xf = rf ./ (miu_1 * gf2sum + hf2);
                
        y_estimate_borders = real(ifft2(xf .* hf));
        y_estimate_borders = y_estimate_borders*alpha + (1-alpha)*y_estimate_borders_old;
        y = Mty + padmask .* y_estimate_borders;
        yf = fft2(y);
        
    end
    
    x = real(ifft2(xf));    
    
    s1 = [diff(x,1,1); x(1,:) - x(end,:)] - d(:,:,1);
    s2 = [diff(x,1,2), x(:,1) - x(:,end)] - d(:,:,2);
    
%     Soft-threshold (isotropic TV)
    V = sqrt(s1.^2 + s2.^2);
    V(V==0) = lambda/miu_1;
    V = max(V - lambda/miu_1, 0)./V;
    v(:,:,1) = s1.*V;
    v(:,:,2) = s2.*V;
       
%     Dual
    d(:,:,1) = - s1 + v(:,:,1);
    d(:,:,2) = - s2 + v(:,:,2);
    
%     Varying penalty parameter
    r_1 = norm(d(:,:,1) - d_old(:,:,1), 'fro');
    r_2 = norm(d(:,:,2) - d_old(:,:,2), 'fro');
    r_1_2 = sqrt(r_1^2 + r_2^2);
    s_1_aux = v(:,:,1) - csi(:,:,1) + d_old(:,:,1);
    s_1 = norm(s_1_aux, 'fro');
    s_2_aux = v(:,:,2) - csi(:,:,2) + d_old(:,:,2);
    s_2 = norm(s_2_aux, 'fro');
    s_1_2 = sqrt(s_1^2 + s_2^2);
    if r_1_2 > 3*s_1_2
        miu_1 = miu_1*2;
        d = d/2;
    elseif s_1_2 > 3*r_1_2
        miu_1 = miu_1/2;
        d = d*2;
    end
        
%     Keep track of time
    if iter == 1
        elapsed(iter) = toc() + pre_elapsed;
    else
        elapsed(iter) = elapsed(iter-1)+toc();
    end
    
%     Ignore boundaries
    x_crop = x(hsize_h+1:end-hsize_h, hsize_w+1:end-hsize_w);
    if mod(iter, show_image) == 0
        sfigure(1);
        imshow(x_crop);
    end
    
%     SSIM, ISNR, SNR
    if mod(iter, compute_quality) == 0
        [snr(iter), ssim(iter), isnr(iter)] = qualitymeasures(original_image, x_crop, observed_image);
        if mod(iter, show_image) == 0
            sfigure(2);
            plot(elapsed(1:iter), isnr(1:iter)), title('ISNR');
            sfigure(3);
            plot(elapsed(1:iter), ssim(1:iter)), title('SSIM');
        end
        if mod(iter, print_status) == 0
            fprintf('t = %5.3f iter = %d  lambda = %8.3e SSIM = %5.3f ISNR = %5.3f  SNR = %5.3f\n', elapsed(iter), iter, lambda, ssim(iter), isnr(iter), snr(iter))
        end
    end
    
%     RMSE with final_image
    if numel(final_image) > 1
        rmse(iter) = norm(x - final_image,'fro')/sqrt(numel(final_image));
        if mod(iter, show_image) == 0
            sfigure(4); 
            semilogy(elapsed(1:iter), rmse(1:iter)), title('RMSE');
        end
        if mod(iter, print_status) == 0
            fprintf('iter = %d lambda = %8.3e RMSE = %2.5e\n', iter, lambda, rmse(iter));
        end
        if (rmse(iter) < min_error)
            snr(iter+1:end) = [];
            isnr(iter+1:end) = [];
            ssim(iter+1:end) = [];
            elapsed(iter+1:end) = [];
            rmse(iter+1:end) = [];
            break
        end
    end
    drawnow
    iter = iter + 1;
    
end
estimated_image = x_crop;
estimated_image_full = x;