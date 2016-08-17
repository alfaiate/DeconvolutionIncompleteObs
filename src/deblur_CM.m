function [ rmse, elapsed, snr, isnr, ssim, estimated_image, estimated_image_full ] = deblur_CM( observed_image, blur_kernel, mask, maxiter, lambda, miu_1, miu_2, original_image, final_image, min_error, show_image, print_status, compute_quality )
%deblur_CM - Corresponds to the CM method discussed in [1]
% 
%   [1] M. Simoes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, 
%        “A Framework for Fast Image Deconvolution with
%        Incomplete Observations,” IEEE Trans. Image Process.,
%        to be published.
% 
% function [ rmse, elapsed, snr, isnr, ssim, estimated_image, estimated_image_full ] 
%           = deblur_AM( observed_image, blur_kernel, mask, maxiter, lambda, miu_1, 
%               miu_2, original_image, final_image, min_error, show_image, 
%               print_status, compute_quality )
% 
% Input: 
% observed_image: blurred image
% blur_kernel: filter
% mask: mask with unobserved pixels
% maxiter: maximum number of iterations
% lambda: regularization parameter
% miu_1: penalization parameter
% miu_2: penalization parameter
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
%
% This script has two steps. 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% I. It starts by initializing the variables used by ADMM-CG.
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

% Properly sized blur filter
id = zeros(size(y));
id(1, 1) = 1;
h = imfilter(id, blur_kernel, 'circular', 'conv');
h = h / sum(h(:)); % Filter calibration

% Initialization
x = y;              % Initialize the estimated image with the blurred one
hf = fft2(h);
hfc = conj(hf);           % Conjugate
HtMtyf = hfc.*fft2(imagemask.*y);

d(:,:,1) = zeros(size(x));
d(:,:,2) = zeros(size(x));

snr = zeros(1, maxiter);
isnr = zeros(1, maxiter);
ssim = zeros(1, maxiter);
elapsed = zeros(1, maxiter);
rmse = zeros(1, maxiter);

iter = 1;

% Warm up tic/toc.
tic();
elapsed(1) = toc();
tic();
elapsed(1) = toc();

% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% II. CM.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 

while iter <= maxiter
    tic();
    
%     Primal
    x_old = x;
    xf = fft2(x);
    HtMtMH_xf = hfc .* fft2(imagemask .* real(ifft2(hf.*xf)));
    x = real(ifft2(xf - miu_1*(HtMtMH_xf - HtMtyf)));
    x = x - miu_1*([d(end,:,1) - d(1,:,1); -diff(d(:,:,1),1,1)] ... 
        + [d(:,end,2) - d(:,1,2), -diff(d(:,:,2),1,2)]);
    
%     Dual
    ad_x = 2*x - x_old;
    s1 = d(:,:,1) + miu_2*[diff(ad_x,1,1); ad_x(1,:) - ad_x(end,:)];
    s2 = d(:,:,2) + miu_2*[diff(ad_x,1,2), ad_x(:,1) - ad_x(:,end)];

%     Soft-threshold (isotropic TV) - conjugate function
    V = sqrt(s1.^2 + s2.^2);
    V = max(1, V/lambda);
    d(:,:,1) = s1./V;
    d(:,:,2) = s2./V;
    
%     Keep track of time
    if iter == 1
        elapsed(iter) = toc();
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