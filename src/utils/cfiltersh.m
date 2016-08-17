function outimage = cfiltersh(image,filter)
%
% Circular-filter an image with a (possibly multiple) filter
%
% Filtering is performed with FFTs, and is therefore circular
%
% Inputs:
%
%   image - input image, grayscale
%
%   filter - filter impulse response (point spread function). Can be of
%   smaller size than the image. Shouldn't be larger than the image, in
%   any dimension. The filter is resized to the image's size, so that
%   filter (without fftshift) is equal to the center part of
%   fftshift(filt), where filt is the filter that's actually used.
%


v = size(image,1);
h = size(image,2);

vf = size(filter,1);
hf = size(filter,2);
nimages = size(filter,3);

if vf > v  || hf > h
  fprintf('Error - Size of filter is larger than size of image\n')
  return
end

imagef = fft2(image);

nkeepv = floor((vf + 1) / 2);
nkeeph = floor((hf + 1) / 2);
nmovev = vf - nkeepv;
nmoveh = hf - nkeeph;

outimage = zeros([size(image) nimages]);

for img = 1:nimages
  
  % Resize the filter
 
  filt = zeros(size(image));
  filt(1:nkeepv,1:nkeeph) = filter(nmovev+1:end,nmoveh+1:end,img);
  filt(end-nmovev+1:end,1:nkeeph) = filter(1:nmovev,nmoveh+1:end,img);
  filt(1:nkeepv,end-nmoveh+1:end) = filter(nmovev+1:end,1:nmoveh,img);
  filt(end-nmovev+1:end,end-nmoveh+1:end) = filter(1:nmovev,1:nmoveh,img);
  
  % Now filter the image
  
  outimage(:,:,img) = real(ifft2(fft2(filt) .* imagef));
  
end