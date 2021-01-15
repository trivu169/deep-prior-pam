clearvars;
n = 500;
m = 500;
%% Without Sides:
%%{
[columnsInImage, rowsInImage] = meshgrid(1:n, 1:m);
centerX = n/2;
centerY = m/2;
radius = (m+n)/4;
%}
%%{
circlePixels = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius.^2;
%}
figure
imshow(circlePixels)
img_size = size(circlePixels);
circlePixels(1:img_size(1)/2, img_size(1)/2:img_size(2)) = 0;
figure
imshow(circlePixels)

circlePixels(img_size(1)/2:img_size(1), 1:img_size(2)/2) = 0;
circlePixels = double(circlePixels);
figure
imshow(circlePixels)
limited_view = 120;
orig_theta = 90 - (180 - limited_view);
mask = imrotate(circlePixels, orig_theta, 'bicubic', 'crop');
mask(1:img_size(1)/2, img_size(2)/2:img_size(2)) = 0;
mask(img_size(1)/2:img_size(1), 1:img_size(2)/2) = 0;
figure
imshow(mask)
mask0 = mask;

%% Optimal Search
min_diff = inf;
theta_min = 0;
filename = ['p0_recon_interp_',num2str(limited_view),'deg.mat'];
load(filename);
fourier = log10(abs(fftshift(fft2(p0_recon_interp))));
% First Search
dt = 0.1;
for theta = 1:dt:360
    binary_mask = imrotate(mask, theta, 'bicubic', 'crop');
    img = fourier .* (1-binary_mask);
    diff = mean(mean(abs(img - fourier)));
    if diff < min_diff
        min_diff = diff;
        theta_min = theta;
    end
end
disp(theta_min)

% Second Search
for theta = theta_min-5*dt:dt/1000:theta_min+5*dt
    binary_mask = imrotate(mask, theta, 'bicubic', 'crop');
    img = fourier .* (1-binary_mask);
    diff = mean(mean(abs(img - fourier)));
    if diff < min_diff
        min_diff = diff;
        theta_min = theta;
    end
end
disp(theta_min)
mask1 = imrotate(mask, theta_min, 'bicubic', 'crop');
figure
imshow(mask1)
%% With Sides:
%%{
[columnsInImage, rowsInImage] = meshgrid(1:2*n, 1:2*m);
centerX = n;
centerY = m;
radius = (m+n)/2;
circlePixels = ones(2*m,2*n);
figure
imshow(circlePixels)
img_size = size(circlePixels);
circlePixels(1:img_size(1)/2, img_size(1)/2:img_size(2)) = 0;
figure
imshow(circlePixels)

circlePixels(img_size(1)/2:img_size(1), 1:img_size(2)/2) = 0;
circlePixels = double(circlePixels);
figure
imshow(circlePixels)
limited_view = 120;
orig_theta = 90 - (180 - limited_view);
mask = imrotate(circlePixels, orig_theta, 'bicubic', 'crop');
mask(1:img_size(1)/2, img_size(2)/2:img_size(2)) = 0;
mask(img_size(1)/2:img_size(1), 1:img_size(2)/2) = 0;
pre_mask = mask;
mask = imrotate(mask, theta_min, 'bicubic', 'crop');
mask = mask(n/2:3*n/2-1, m/2:3*n/2-1);
figure
imshow(mask)
mask2 = mask;
%}

%% Gaussian Mask 1:
axis = 'y';
[m,n] = size(binary_mask);
gaussian_profile = zeros(m,n);
center = size(binary_mask)/2;
sig = 0.1*m;
mean = center(1);
input = 1:m;
gaussian_vector = gaussmf(input,[sig mean]);
if axis == 'x'
    for i = 1:m
        gaussian_profile(i,:) = gaussian_vector;
    end
elseif axis == 'y'
    for j = 1:n
        gaussian_profile(:,j) = gaussian_vector;
    end
end
figure
imshow(gaussian_profile)
title('Gaussian Profile')
gauss_mask1 = imrotate(mask0, (90 - orig_theta)/2, 'bicubic', 'crop').*gaussian_profile;
gauss_mask1 = imrotate(gauss_mask1, -(90 - orig_theta)/2, 'bicubic', 'crop');
gauss_mask1 = imrotate(gauss_mask1, theta_min, 'bicubic', 'crop');
gauss_mask1 = gauss_mask1 + abs(gauss_mask1);
gauss_mask1 = gauss_mask1/max(max(gauss_mask1));
figure
imshow(gauss_mask1)
title('Gaussian Mask 1')

%% Gaussian Mask 1:
axis = 'y';
[m,n] = size(binary_mask);
gaussian_profile = zeros(2*m,2*n);
center = size(binary_mask)/2;
sig = 0.1*(m);
mean = m;
input = 1:2*m;
gaussian_vector = gaussmf(input,[sig mean]);
if axis == 'x'
    for i = 1:2*m
        gaussian_profile(i,:) = gaussian_vector;
    end
elseif axis == 'y'
    for j = 1:2*n
        gaussian_profile(:,j) = gaussian_vector;
    end
end
figure
imshow(gaussian_profile)
title('Gaussian Profile')
gauss_mask2 = imrotate(pre_mask, (90 - orig_theta)/2, 'bicubic', 'crop').*gaussian_profile;
gauss_mask2 = imrotate(gauss_mask2, -(90 - orig_theta)/2, 'bicubic', 'crop');
gauss_mask2 = imrotate(gauss_mask2, theta_min, 'bicubic', 'crop');
gauss_mask2 = gauss_mask2(n/2:3*n/2-1, m/2:3*n/2-1);
gauss_mask2 = gauss_mask2 + abs(gauss_mask2);
gauss_mask2 = gauss_mask2/max(max(gauss_mask2));
figure
imshow(gauss_mask2)
title('Gaussian Mask 2')
%% Save Image
imwrite(mask1, 'ideal_mask_circ.png');
imwrite(mask2, 'ideal_mask_corner.png');
imwrite(gauss_mask1, 'gauss_mask_circ.png');
imwrite(gauss_mask2, 'gauss_mask_corner.png');