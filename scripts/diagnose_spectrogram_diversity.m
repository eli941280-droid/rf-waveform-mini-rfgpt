function diagnose_spectrogram_diversity(spectrogramDir)
%DIAGNOSE_SPECTROGRAM_DIVERSITY Diagnose visual diversity of spectrogram PNGs.
%
% Usage:
%   diagnose_spectrogram_diversity
%   diagnose_spectrogram_diversity("data/spectrograms")
%   diagnose_spectrogram_diversity("data_wlan/spectrograms")

    clc; close all;

if nargin < 1 || isempty(spectrogramDir)
    spectrogramDir = fullfile("data", "spectrograms");
end

spectrogramDir = char(spectrogramDir);
outputTag = get_output_tag(spectrogramDir);
outputDir = fullfile(pwd, "outputs");

if ~exist(outputDir, "dir")
    [mkdirOk, mkdirMsg] = mkdir(outputDir);
    if ~mkdirOk
        error("Cannot create output directory: %s. %s", outputDir, mkdirMsg);
    end
end

writeTestPath = fullfile(outputDir, ".write_test.tmp");
[writeTestFid, writeTestMsg] = fopen(writeTestPath, "w");
if writeTestFid < 0
    error("Output directory is not writable: %s. %s", outputDir, writeTestMsg);
end
fclose(writeTestFid);
delete(writeTestPath);

files = dir(fullfile(spectrogramDir, "*.png"));
if isempty(files)
    error("No PNG files found in %s", spectrogramDir);
end

[~, sortIdx] = sort({files.name});
files = files(sortIdx);

numImages = min(100, numel(files));
if numImages < 2
    error("At least two PNG files are required for diversity diagnosis.");
end

if numImages < 100
    warning("Only found %d PNG files in %s.", numImages, spectrogramDir);
end

files = files(1:numImages);
montageCols = ceil(sqrt(numImages));
montageRows = ceil(numImages / montageCols);
imageSize = [];
imageVectors = [];
montageImage = [];

for i = 1:numImages
    imgPath = fullfile(files(i).folder, files(i).name);
    img = imread(imgPath);

    if ndims(img) == 3
        img = double(img(:, :, 1)) * 0.2989 + ...
            double(img(:, :, 2)) * 0.5870 + ...
            double(img(:, :, 3)) * 0.1140;
    else
        img = double(img);
    end

    imgMin = min(img(:));
    imgMax = max(img(:));
    if imgMax > imgMin
        img = (img - imgMin) / (imgMax - imgMin);
    else
        img = zeros(size(img));
    end

    if i == 1
        imageSize = size(img);
        imageVectors = zeros(numImages, numel(img));
        tileHeight = imageSize(1);
        tileWidth = imageSize(2);
        montageImage = repmat(uint8(255), ...
            montageRows * tileHeight, montageCols * tileWidth);
    elseif ~isequal(size(img), imageSize)
        if exist("imresize", "file") == 2
            img = imresize(img, imageSize);
        else
            error("Image size mismatch and imresize is unavailable: %s", imgPath);
        end
    end

    rowIdx = floor((i - 1) / montageCols) + 1;
    colIdx = mod(i - 1, montageCols) + 1;

    r1 = (rowIdx - 1) * tileHeight + 1;
    r2 = rowIdx * tileHeight;
    c1 = (colIdx - 1) * tileWidth + 1;
    c2 = colIdx * tileWidth;

    montageImage(r1:r2, c1:c2) = uint8(round(255 * min(max(img, 0), 1)));
    imageVectors(i, :) = img(:).';
end

montagePath = fullfile(outputDir, char("diversity_" + outputTag + "_montage.png"));
imwrite(montageImage, montagePath);

vectorNorms = sqrt(sum(imageVectors.^2, 2));
normalizedVectors = imageVectors ./ max(vectorNorms, eps);
similarityMatrix = normalizedVectors * normalizedVectors.';
pairMask = triu(true(numImages), 1);
pairSimilarities = similarityMatrix(pairMask);

meanSimilarity = mean(pairSimilarities);
medianSimilarity = median(pairSimilarities);
maxSimilarity = max(pairSimilarities);
minSimilarity = min(pairSimilarities);

centeredVectors = imageVectors - mean(imageVectors, 1);
gramMatrix = centeredVectors * centeredVectors.';
gramMatrix = (gramMatrix + gramMatrix.') / 2;
[eigVectors, eigValuesMatrix] = eig(gramMatrix);
[eigValues, eigOrder] = sort(diag(eigValuesMatrix), "descend");
eigVectors = eigVectors(:, eigOrder);

scores = eigVectors(:, 1:2) * diag(sqrt(max(eigValues(1:2), 0)));

pcaPath = fullfile(outputDir, char("diversity_" + outputTag + "_pca.png"));
fig = figure("Visible", "off");
scatter(scores(:, 1), scores(:, 2), 36, "filled");
grid on;
xlabel("PC1");
ylabel("PC2");
title("Spectrogram Diversity PCA");
print(fig, pcaPath, "-dpng", "-r150");
close(fig);

fprintf("Diagnosed %d spectrogram PNG files from %s\n", numImages, spectrogramDir);
fprintf("Montage saved to: %s\n", montagePath);
fprintf("PCA plot saved to: %s\n", pcaPath);
fprintf("Mean cosine similarity   : %.6f\n", meanSimilarity);
fprintf("Median cosine similarity : %.6f\n", medianSimilarity);
fprintf("Max cosine similarity    : %.6f\n", maxSimilarity);
fprintf("Min cosine similarity    : %.6f\n", minSimilarity);

if meanSimilarity > 0.95
    conclusion = char([26679 26412 39640 24230 30456 20284 ...
        65292 22810 26679 24615 19981 36275]);
elseif meanSimilarity >= 0.85
    conclusion = char([26679 26412 20013 24230 30456 20284 ...
        65292 38656 35201 22686 24378 32467 26500 38543 26426 21270]);
else
    conclusion = char([22810 26679 24615 21021 27493 21487 25509 21463]);
end

fprintf("%s%s\n", char([35786 26029 32467 35770 65306]), conclusion);

end

function outputTag = get_output_tag(spectrogramDir)
    dirText = lower(char(spectrogramDir));
    normalizedText = strrep(dirText, filesep, "/");
    if contains(normalizedText, "data_wlan") || contains(normalizedText, "wlan")
        outputTag = "wlan";
    elseif contains(normalizedText, "5g") || endsWith(normalizedText, "data/spectrograms")
        outputTag = "5g";
    else
        [~, parentName] = fileparts(fileparts(char(spectrogramDir)));
        [~, folderName] = fileparts(char(spectrogramDir));
        if ~isempty(parentName) && ~strcmp(parentName, filesep)
            outputTag = string(parentName);
        else
            outputTag = string(folderName);
        end
        outputTag = regexprep(outputTag, "[^A-Za-z0-9_]", "_");
        if strlength(outputTag) == 0
            outputTag = "custom";
        end
    end
end
