function b = padimage(a, padSize, padVal, direction)
% PADIMAGE  Pad image.
%
% Usage is identical to padarray but with additional options:
% 'linearInterpolation'
%
% Miguel Simões, 2012

if ischar(padVal)   
    switch padVal
        case 'linearInterpolation'
            % Adapted from Chakrabarti, A., & Zickler, T. (2012). Fast Deconvolution with Color Constraints on Gradients, 1–3. Retrieved from ftp://ftp.deas.harvard.edu/pub/techreports/tr-06-12.pdf
            % It now supports non-squared blurs, i.e., padSize can be an
            % array
            
            b = zeros(size(a) + 2*[padSize(1) padSize(2)]); % note that they've been swapped
            fadex = linspace(0, 1, 2*padSize(2));
            fadey = linspace(0, 1, 2*padSize(1))';
            wtx = repmat(fadex, [size(a, 1) 1]);
            wty = repmat(fadey, [1 size(b, 2)]);
            
            yi = a;
            
            px = (1-wtx) .* repmat(yi(:,end), [1 2*padSize(2)]) + (wtx) .* repmat(yi(:,1), [1 2*padSize(2)]);
            yi = [px(:, padSize(2)+1:end) yi px(:, 1:padSize(2))];
            
            py = (1-wty) .* repmat(yi(end,:), [2*padSize(1) 1]) + (wty) .* repmat(yi(1,:), [2*padSize(1) 1]);
            yi = [py(padSize(1)+1:end, :); yi; py(1:padSize(1), :)];
            
            b = yi;
                      
        otherwise
            
            b = padarray(a, padSize, padVal);
            
    end    
else
    
    b = padarray(a, padSize, padVal); % not very generic...
    
end