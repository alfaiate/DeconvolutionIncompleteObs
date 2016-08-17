function [snr, ssim, isnr, mssim, x0a, xa, ya] = qualitymeasures(x0, x, y, clip, normalize)
% Compute the ISNR, SNR and SSIM of the deblurred image x relative to the original sharp image x0, the blurred image being y.
%
% If y is not given, only the SNR and SSIM are computed.
%
% The images x and x0 are first aligned relative to each other by shifting by the number of pixels that corresponds to the maximum of their
% correlation. The quality measures are computed using only the area that is common to both images.
%
% If y is given, it is also aligned with x0 in the same way, and its SNR also takes into account only the area that is common to y and x0.
%
% clip - Clip the three images to the range [0,1]. The clipping is performed before normalization. Default: do not clip.
%
% normalize - Adjust the contrast and brightness of x and y for the best fit to x0. Default: do not normalize.



% Clip if requested

if nargin >= 4 && clip
	x = max(0, min(1, x));
	if exist('y', 'var')
		y = max(0, min(1, y));
	end
end



% Find SNR of x
% First align the images

corr = real(ifft2(fft2(x0) .* conj(fft2(x))));		% Find the circular correlation between the two images (the linear correlation takes too long)

% Find the position of the maximum
[lixo,pos] = max(corr(:));
position(1) = mod(pos, size(x0,1)) - 1;
position(2) = floor(pos / size(x0,1));

% Correct position of maximum regarding wraparound of negative values

for i=1:2
	if position(i) > size(x0,i) / 2
		position(i) = position(i) - size(x0,i); %#ok<AGROW>
	end
end

% Crop both x0 and x so that only the common area remains

if position(1) < 0
	x0a = x0(1:end+position(1),:);
	xa = x(1-position(1):end,:);
else
	x0a = x0(1+position(1):end,:);
	xa = x(1:end-position(1),:);
end

if position(2) < 0
	x0a = x0a(:,1:end+position(2));
	xa = xa(:,1-position(2):end);
else
	x0a = x0a(:,1+position(2):end);
	xa = xa(:,1:end-position(2));
end

% Normalize if requested

if nargin >= 5 && normalize
	param = polyfit(xa(:), x0a(:), 1);
	xa = param(1) * xa + param(2);
end

% xa = max(min(x0a(:)), min(max(x0a(:)), xa));

% sfigure(5); imagesc(x0a-xa), colormap gray(256)

% Now find the SNR

snr = 20 * log10(norm(x0a,'fro') / norm((xa-x0a),'fro'));




% Find the SSIM

% ssim = ssim_index(min(1,max(0,x))*255, min(1,max(0,x0))*255);
ssim = ssim_index(xa*255, x0a*255);

% Find the MSSIM
% mssim = metrix_mssim(xa*255, x0a*255);


% Find the ISNR if y is given

if nargin >= 3
	
	% Find the SNR of y
	% First align the images
	
	corr = real(ifft2(fft2(x0) .* conj(fft2(y))));		% Find the circular correlation between the two images (the linear correlation takes too long)
	
	% Find the position of the maximum
	[lixo,pos] = max(corr(:));
	position(1) = mod(pos, size(x0,1)) - 1;
	position(2) = floor(pos / size(x0,1));
	
	% Correct position of maximum regarding wraparound of negative values
	
	for i=1:2
		if position(i) > size(x0,i) / 2
			position(i) = position(i) - size(x0,i); %#ok<AGROW>
		end
	end
	
	% Crop both x0 and y so that only the common area remains
	
	if position(1) < 0
		x0a2 = x0(1:end+position(1),:);
		ya = y(1-position(1):end,:);
	else
		x0a2 = x0(1+position(1):end,:);
		ya = y(1:end-position(1),:);
	end
	
	if position(2) < 0
		x0a2 = x0a2(:,1:end+position(2));
		ya = ya(:,1-position(2):end);
	else
		x0a2 = x0a2(:,1+position(2):end);
		ya = ya(:,1:end-position(2));
	end
	
	% Normalize if requested
	
	if nargin >= 5 && normalize
		param = polyfit(ya(:), x0a2(:), 1);
		ya = param(1) * ya + param(2);
	end
	
% 	sfigure(6); imagesc(x0a2-ya), colormap gray(256)

	% Now find the ISNR
	
	snry = 20 * log10(norm(x0a2,'fro') / norm((ya-x0a2),'fro'));
	isnr = snr - snry;
	
end


