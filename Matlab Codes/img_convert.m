clearvars;
%{
load('p0_recon_interp_full.mat')
fourier = log10(abs(fftshift(fft2(mat2gray(p0_recon_interp)))));
figure
imshow(fourier)
p0_recon_interp_full = uint8(255 * mat2gray(fourier));
load('p0_recon_interp_120deg.mat')
fourier = log10(abs(fftshift(fft2(mat2gray(p0_recon_interp)))));
figure
imshow(fourier)
p0_recon_interp_down = uint8(255 * mat2gray(fourier));

binary_mask = imread('best_mask.png');
binary_mask = uint8(255 * mat2gray(1-binary_mask));

rgb = cat(3, p0_recon_interp_full, p0_recon_interp_down, binary_mask);
imwrite(p0_recon_interp_full,'fourier_full.png')
imwrite(p0_recon_interp_down,'fourier_limited.png')
imwrite(p0_recon_interp_down>80,'fourier_limited_thres.png')
imwrite(p0_recon_interp_full>80,'fourier_full_thres.png')
imwrite(binary_mask,'best_mask_inv.png')
figure
imshow(rgb(:,:,1))
figure
imshow(rgb(:,:,2))
imwrite(rgb,'combined_image_sim.png')
%}
%% Ideal Mask Circ:
binary_mask = imread('ideal_mask_circ.png');

binary_mask = uint8(255 * mat2gray(1-binary_mask));

imwrite(binary_mask,'ideal_mask_circ.png')
load('p0_recon_interp_full.mat')
p0_recon_interp_full = uint8(255 * mat2gray(p0_recon_interp));
load('p0_recon_interp_120deg.mat')
p0_recon_interp_down = uint8(255 * mat2gray(p0_recon_interp));
rgb = cat(3, p0_recon_interp_full, p0_recon_interp_down, binary_mask);
imwrite(rgb,'combined_spatial_ideal_circ.png')

%% Ideal Mask Corners:
binary_mask = imread('ideal_mask_corner.png');

binary_mask = uint8(255 * mat2gray(1-binary_mask));

imwrite(binary_mask,'ideal_mask_corner_inv.png')
load('p0_recon_interp_full.mat')
p0_recon_interp_full = uint8(255 * mat2gray(p0_recon_interp));
load('p0_recon_interp_120deg.mat')
p0_recon_interp_down = uint8(255 * mat2gray(p0_recon_interp));
rgb = cat(3, p0_recon_interp_full, p0_recon_interp_down, binary_mask);
imwrite(rgb,'combined_spatial_ideal_corner.png')

%% Gauss Mask Ideal:
binary_mask = imread('gauss_mask_circ.png');

binary_mask = uint8(255 * (1-mat2gray(binary_mask)));
% figure
% imshow(binary_mask)
imwrite(binary_mask,'gauss_mask_circ_inv.png')
load('p0_recon_interp_full.mat')
p0_recon_interp_full = uint8(255 * mat2gray(p0_recon_interp));
load('p0_recon_interp_120deg.mat')
p0_recon_interp_down = uint8(255 * mat2gray(p0_recon_interp));
rgb = cat(3, p0_recon_interp_full, p0_recon_interp_down, binary_mask);
imwrite(rgb,'combined_spatial_gauss_circ.png')

%% Gauss Mask Corners:
binary_mask = imread('gauss_mask_corner.png');

binary_mask = uint8(255 * (1-mat2gray(binary_mask)));

imwrite(binary_mask,'gauss_mask_corner_inv.png')
load('p0_recon_interp_full.mat')
p0_recon_interp_full = uint8(255 * mat2gray(p0_recon_interp));
load('p0_recon_interp_120deg.mat')
p0_recon_interp_down = uint8(255 * mat2gray(p0_recon_interp));
rgb = cat(3, p0_recon_interp_full, p0_recon_interp_down, binary_mask);
imwrite(rgb,'combined_spatial_gauss_corner.png')