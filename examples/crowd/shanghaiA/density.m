function density_map = density(image_path, position_head_path, k)
%function: generate the density map
%@params:
%img_path: path of input image
%position_head_path: path of ground truth containing the position of person's head
%k: the k nearest neighbor
if nargin < 3
    k = 7;
end
image = imread(image_path);
position_head_struct = load(position_head_path);
position_head = position_head_struct.image_info{1,1}.location;
distance_mat = distance(position_head, k);
density_map = zeros(size(image,1), size(image,2));
for pid = 1:size(position_head,1)
    var = 0.3 * mean(distance_mat(pid, :));
    ph = [floor(position_head(pid,2)), floor(position_head(pid,1))];
    dh = norm2d(image, [var, var], [ph(1),ph(2)]);
    dh = dh./sum(sum(dh));
    density_map = dh + density_map;
end
end

function dense = norm2d(input, sigma, center)
%function: generate 2d normal distribution
%@params:
%input: input size
%sigma: [sigmay, sigmax]
%center: [centerx, centery]
    gsize = size(input);
    [X1, X2] = meshgrid(1:gsize(1), 1:gsize(2));
    Sigma = zeros(size(sigma,2));
    for i = 1:size(sigma,2)
        Sigma(i,i) = sigma(i)^2;
    end
    dense = mvnpdf([X1(:) X2(:)], center, Sigma);
    dense = reshape(dense, gsize(2), gsize(1))';
end

function distance_matrix = distance(position_head, k)
%function: caculate the distance matrix
%@params:
%k: the k nearest neighbor
head_num = size(position_head, 1);
distance_matrix = zeros(head_num, head_num);
for i = 1:head_num
    for j = 1:head_num
        distance_matrix(i, j) = sum((position_head(i,:)-position_head(j,:)).^2);
    end
end
distance_matrix = sort(distance_matrix, 2);
distance_matrix = sqrt(distance_matrix(:, 1:k));
end