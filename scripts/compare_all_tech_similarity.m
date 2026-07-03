clc; clear; close all;

spectrogramDir = fullfile("data_all", "spectrograms");
outputDir = "outputs";
outputTxtPath = fullfile(outputDir, "compare_all_tech_similarity.txt");
outputHeatmapPath = fullfile(outputDir, "compare_all_tech_similarity_heatmap.png");

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

techs = {
    "5G NR", "5g_nr"
    "LTE", "lte"
    "UMTS", "umts"
    "WLAN", "wlan"
    "DVB-S2", "dvbs2"
    "Bluetooth", "bluetooth"
};

maxImagesPerTech = 100;
vectorsByTech = cell(size(techs, 1), 1);

for i = 1:size(techs, 1)
    pattern = char(techs{i, 2} + "_*.png");
    vectorsByTech{i} = load_image_vectors(spectrogramDir, pattern, maxImagesPerTech);
end

meanMatrix = zeros(size(techs, 1));
medianMatrix = zeros(size(techs, 1));
maxMatrix = zeros(size(techs, 1));
minMatrix = zeros(size(techs, 1));

for i = 1:size(techs, 1)
    for j = 1:size(techs, 1)
        stats = pairwise_cosine_stats(vectorsByTech{i}, vectorsByTech{j}, i == j);
        meanMatrix(i, j) = stats.mean;
        medianMatrix(i, j) = stats.median;
        maxMatrix(i, j) = stats.max;
        minMatrix(i, j) = stats.min;
    end
end

print_and_write_report(outputTxtPath, techs, meanMatrix, medianMatrix, maxMatrix, minMatrix);
plot_heatmap(outputHeatmapPath, techs, meanMatrix);

fprintf("Saved all-tech similarity report to: %s\n", outputTxtPath);
fprintf("Saved all-tech similarity heatmap to: %s\n", outputHeatmapPath);

function vectors = load_image_vectors(imageDir, pattern, maxImages)
    files = dir(fullfile(imageDir, pattern));
    if isempty(files)
        error("No PNG files found for pattern %s in %s", pattern, imageDir);
    end

    [~, sortIdx] = sort({files.name});
    files = files(sortIdx);
    numImages = min(maxImages, numel(files));

    targetSize = [512 512];
    vectors = zeros(numImages, prod(targetSize));

    for i = 1:numImages
        imgPath = fullfile(files(i).folder, files(i).name);
        img = read_normalized_gray_image(imgPath);
        if ~isequal(size(img), targetSize)
            img = imresize(img, targetSize);
        end
        vectors(i, :) = img(:).';
    end
end

function img = read_normalized_gray_image(imgPath)
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
end

function stats = pairwise_cosine_stats(vectorsA, vectorsB, isIntraClass)
    normalizedA = normalize_rows(vectorsA);
    normalizedB = normalize_rows(vectorsB);
    similarityMatrix = normalizedA * normalizedB.';
    if isIntraClass
        pairMask = triu(true(size(similarityMatrix)), 1);
        similarities = similarityMatrix(pairMask);
    else
        similarities = similarityMatrix(:);
    end
    stats = struct;
    stats.mean = mean(similarities);
    stats.median = median(similarities);
    stats.max = max(similarities);
    stats.min = min(similarities);
end

function normalizedVectors = normalize_rows(vectors)
    vectorNorms = sqrt(sum(vectors.^2, 2));
    normalizedVectors = vectors ./ max(vectorNorms, eps);
end

function print_and_write_report(outputTxtPath, techs, meanMatrix, medianMatrix, maxMatrix, minMatrix)
    fid = fopen(outputTxtPath, "w");
    if fid < 0
        error("Cannot open report file: %s", outputTxtPath);
    end
    cleanupObj = onCleanup(@() fclose(fid));

    fprintf("Mean cosine similarity matrix:\n");
    fprintf(fid, "Mean cosine similarity matrix:\n");
    print_matrix(techs, meanMatrix, 1);
    print_matrix(techs, meanMatrix, fid);

    fprintf("\nMedian cosine similarity matrix:\n");
    fprintf(fid, "\nMedian cosine similarity matrix:\n");
    print_matrix(techs, medianMatrix, 1);
    print_matrix(techs, medianMatrix, fid);

    fprintf("\nMax cosine similarity matrix:\n");
    fprintf(fid, "\nMax cosine similarity matrix:\n");
    print_matrix(techs, maxMatrix, 1);
    print_matrix(techs, maxMatrix, fid);

    fprintf("\nMin cosine similarity matrix:\n");
    fprintf(fid, "\nMin cosine similarity matrix:\n");
    print_matrix(techs, minMatrix, 1);
    print_matrix(techs, minMatrix, fid);

    intraMeans = diag(meanMatrix);
    interMask = ~eye(size(meanMatrix));
    meanIntra = mean(intraMeans);
    meanInter = mean(meanMatrix(interMask));

    conclusion = "Inter-class similarity is close to intra-class similarity; WTR may need stronger technology-specific structure.";
    if meanIntra - meanInter >= 0.05
        conclusion = "Inter-class similarity is clearly lower than intra-class similarity; WTR has visible technology separation.";
    end

    fprintf("\nAverage intra-class mean similarity: %.6f\n", meanIntra);
    fprintf("Average inter-class mean similarity: %.6f\n", meanInter);
    fprintf("Diagnostic conclusion: %s\n", conclusion);

    fprintf(fid, "\nAverage intra-class mean similarity: %.6f\n", meanIntra);
    fprintf(fid, "Average inter-class mean similarity: %.6f\n", meanInter);
    fprintf(fid, "Diagnostic conclusion: %s\n", conclusion);
end

function print_matrix(techs, matrixValue, fid)
    fprintf(fid, "%14s", "");
    for j = 1:size(techs, 1)
        fprintf(fid, "%14s", char(techs{j, 1}));
    end
    fprintf(fid, "\n");
    for i = 1:size(techs, 1)
        fprintf(fid, "%14s", char(techs{i, 1}));
        for j = 1:size(techs, 1)
            fprintf(fid, "%14.6f", matrixValue(i, j));
        end
        fprintf(fid, "\n");
    end
end

function plot_heatmap(outputHeatmapPath, techs, meanMatrix)
    fig = figure("Visible", "off");
    imagesc(meanMatrix);
    axis image;
    colorbar;
    caxis([0 1]);
    set(gca, "XTick", 1:size(techs, 1), "XTickLabel", techs(:, 1));
    set(gca, "YTick", 1:size(techs, 1), "YTickLabel", techs(:, 1));
    xtickangle(45);
    title("All-Technology Mean Cosine Similarity");
    print(fig, outputHeatmapPath, "-dpng", "-r150");
    close(fig);
end
