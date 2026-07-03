clc; clear; close all;

fiveGDir = fullfile("data", "spectrograms");
wlanDir = fullfile("data_wlan", "spectrograms");
outputDir = "outputs";
outputTxtPath = fullfile(outputDir, "compare_5g_wlan_similarity.txt");
outputBarPath = fullfile(outputDir, "compare_5g_wlan_similarity_bar.png");

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

fiveGVectors = load_image_vectors(fiveGDir, 100);
wlanVectors = load_image_vectors(wlanDir, 100);

fiveGIntra = pairwise_cosine_stats(fiveGVectors, fiveGVectors, true);
wlanIntra = pairwise_cosine_stats(wlanVectors, wlanVectors, true);
interStats = pairwise_cosine_stats(fiveGVectors, wlanVectors, false);

diagnosis = make_diagnosis(fiveGIntra.mean, wlanIntra.mean, interStats.mean);

print_results(fiveGIntra, wlanIntra, interStats, diagnosis);
write_results(outputTxtPath, fiveGIntra, wlanIntra, interStats, diagnosis);
plot_bar(outputBarPath, fiveGIntra.mean, wlanIntra.mean, interStats.mean);

fprintf("Saved comparison report to: %s\n", outputTxtPath);
fprintf("Saved comparison bar chart to: %s\n", outputBarPath);

function vectors = load_image_vectors(imageDir, maxImages)
    files = dir(fullfile(imageDir, "*.png"));
    if isempty(files)
        error("No PNG files found in %s", imageDir);
    end

    [~, sortIdx] = sort({files.name});
    files = files(sortIdx);

    numImages = min(maxImages, numel(files));
    if numImages < 2
        error("At least two PNG files are required in %s.", imageDir);
    end

    if numImages < maxImages
        warning("Only found %d PNG files in %s.", numImages, imageDir);
    end

    targetSize = [512 512];
    vectors = zeros(numImages, prod(targetSize));

    for i = 1:numImages
        imgPath = fullfile(files(i).folder, files(i).name);
        img = read_normalized_gray_image(imgPath);

        if ~isequal(size(img), targetSize)
            if exist("imresize", "file") == 2
                img = imresize(img, targetSize);
            else
                error("Image size mismatch and imresize is unavailable: %s", imgPath);
            end
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

function diagnosis = make_diagnosis(fiveGMean, wlanMean, interMean)
    intraMean = mean([fiveGMean, wlanMean]);
    margin = intraMean - interMean;

    if margin >= 0.05
        diagnosis = "Inter-class similarity is clearly lower than intra-class similarity, so 5G NR and WLAN have visible separation.";
    else
        diagnosis = "Inter-class similarity is close to intra-class similarity, so 5G NR and WLAN are not visually separated enough under the current image pipeline.";
    end
end

function print_results(fiveGIntra, wlanIntra, interStats, diagnosis)
    fprintf("5G NR intra-class cosine similarity:\n");
    print_stats(fiveGIntra);

    fprintf("\nWLAN intra-class cosine similarity:\n");
    print_stats(wlanIntra);

    fprintf("\n5G NR vs WLAN inter-class cosine similarity:\n");
    print_stats(interStats);

    fprintf("\nDiagnostic conclusion:\n%s\n", diagnosis);
end

function write_results(outputTxtPath, fiveGIntra, wlanIntra, interStats, diagnosis)
    fid = fopen(outputTxtPath, "w");
    if fid < 0
        error("Cannot open output report file: %s", outputTxtPath);
    end

    fprintf(fid, "5G NR intra-class cosine similarity:\n");
    write_stats(fid, fiveGIntra);

    fprintf(fid, "\nWLAN intra-class cosine similarity:\n");
    write_stats(fid, wlanIntra);

    fprintf(fid, "\n5G NR vs WLAN inter-class cosine similarity:\n");
    write_stats(fid, interStats);

    fprintf(fid, "\nDiagnostic conclusion:\n%s\n", diagnosis);
    fclose(fid);
end

function print_stats(stats)
    fprintf("  mean   : %.6f\n", stats.mean);
    fprintf("  median : %.6f\n", stats.median);
    fprintf("  max    : %.6f\n", stats.max);
    fprintf("  min    : %.6f\n", stats.min);
end

function write_stats(fid, stats)
    fprintf(fid, "  mean   : %.6f\n", stats.mean);
    fprintf(fid, "  median : %.6f\n", stats.median);
    fprintf(fid, "  max    : %.6f\n", stats.max);
    fprintf(fid, "  min    : %.6f\n", stats.min);
end

function plot_bar(outputBarPath, fiveGMean, wlanMean, interMean)
    fig = figure("Visible", "off");
    bar([fiveGMean, wlanMean, interMean]);
    ylim([0 1]);
    grid on;
    set(gca, "XTickLabel", {"5G intra", "WLAN intra", "5G-WLAN inter"});
    ylabel("Mean cosine similarity");
    title("5G NR vs WLAN Spectrogram Similarity");
    print(fig, outputBarPath, "-dpng", "-r150");
    close(fig);
end
