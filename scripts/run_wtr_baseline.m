clc; clear; close all;

rootDir = "data_all";
splitDir = fullfile(rootDir, "splits");
outputDir = "outputs";

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

trainPath = fullfile(splitDir, "wtr_train.jsonl");
valPath = fullfile(splitDir, "wtr_val.jsonl");
testPath = fullfile(splitDir, "wtr_test.jsonl");

requiredFiles = [trainPath, valPath, testPath];
for i = 1:numel(requiredFiles)
    if ~exist(requiredFiles(i), "file")
        error("Required WTR split file not found: %s", requiredFiles(i));
    end
end

labels = ["5G NR"; "LTE"; "UMTS"; "WLAN"; "DVB-S2"; "Bluetooth"];
targetSize = [128 128];

trainSet = load_wtr_split(trainPath);
valSet = load_wtr_split(valPath);
testSet = load_wtr_split(testPath);

fprintf("Loaded WTR splits: train=%d, val=%d, test=%d\n", ...
    numel(trainSet.ids), numel(valSet.ids), numel(testSet.ids));

trainFeatures = normalize_rows(extract_image_features(trainSet.images, targetSize));
valFeatures = normalize_rows(extract_image_features(valSet.images, targetSize));
testFeatures = normalize_rows(extract_image_features(testSet.images, targetSize));

centroids = build_centroids(trainFeatures, trainSet.labels, labels);

valPred.centroid = predict_centroid(valFeatures, centroids, labels);
valPred.knn1 = predict_knn(valFeatures, trainFeatures, trainSet.labels, 1);
valPred.knn5 = predict_knn(valFeatures, trainFeatures, trainSet.labels, 5);

testPred.centroid = predict_centroid(testFeatures, centroids, labels);
testPred.knn1 = predict_knn(testFeatures, trainFeatures, trainSet.labels, 1);
testPred.knn5 = predict_knn(testFeatures, trainFeatures, trainSet.labels, 5);

methodNames = ["centroid", "knn1", "knn5"];
methodPriority = ["knn1", "centroid", "knn5"];
valAcc = zeros(numel(methodNames), 1);
testAcc = zeros(numel(methodNames), 1);

for i = 1:numel(methodNames)
    method = methodNames(i);
    valAcc(i) = mean(valPred.(char(method)) == valSet.labels);
    testAcc(i) = mean(testPred.(char(method)) == testSet.labels);
end

bestValAcc = max(valAcc);
candidateMethods = methodNames(abs(valAcc - bestValAcc) < 1e-12);
selectedMethod = candidateMethods(1);
for i = 1:numel(methodPriority)
    if any(candidateMethods == methodPriority(i))
        selectedMethod = methodPriority(i);
        break;
    end
end
selectedTestPred = testPred.(char(selectedMethod));

reportPath = fullfile(outputDir, "wtr_baseline_report.txt");
predictionPath = fullfile(outputDir, "wtr_baseline_predictions.csv");
confusionPath = fullfile(outputDir, "wtr_baseline_confusion.png");

write_report(reportPath, labels, trainSet, valSet, testSet, methodNames, methodPriority, ...
    valAcc, testAcc, selectedMethod, selectedTestPred);
write_predictions(predictionPath, testSet, testPred, selectedMethod);
plot_confusion(confusionPath, labels, testSet.labels, selectedTestPred, selectedMethod);

fprintf("\nWTR baseline complete.\n");
fprintf("Selected method by validation accuracy and fixed tie-break: %s\n", selectedMethod);
fprintf("Selected test accuracy: %.2f%%\n", 100 * mean(selectedTestPred == testSet.labels));
fprintf("Report: %s\n", reportPath);
fprintf("Predictions: %s\n", predictionPath);
fprintf("Confusion matrix: %s\n", confusionPath);

function dataset = load_wtr_split(path)
    fid = fopen(path, "r");
    if fid < 0
        error("Cannot open WTR split: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    dataset.ids = strings(0, 1);
    dataset.images = strings(0, 1);
    dataset.labels = strings(0, 1);

    lineNumber = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        lineNumber = lineNumber + 1;
        line = strtrim(line);
        if isempty(line)
            continue;
        end

        try
            rec = jsondecode(line);
        catch ME
            warning("Skipping %s line %d: %s", path, lineNumber, ME.message);
            continue;
        end

        dataset.ids(end + 1, 1) = get_field_as_string(rec, "id", ""); %#ok<AGROW>
        dataset.images(end + 1, 1) = normalize_path(get_field_as_string(rec, "image", "")); %#ok<AGROW>
        dataset.labels(end + 1, 1) = get_field_as_string(rec, "answer", "unknown"); %#ok<AGROW>
    end
end

function features = extract_image_features(imagePaths, targetSize)
    numImages = numel(imagePaths);
    rawDim = prod(targetSize);
    profileDim = targetSize(1) + targetSize(2);
    summaryDim = 6;
    features = zeros(numImages, rawDim + profileDim + summaryDim, "single");

    for i = 1:numImages
        if ~exist(imagePaths(i), "file")
            error("Image file not found: %s", imagePaths(i));
        end

        img = imread(imagePaths(i));
        gray = normalize_gray(img);
        if size(gray, 1) ~= targetSize(1) || size(gray, 2) ~= targetSize(2)
            gray = imresize(gray, targetSize);
        end

        freqProfile = mean(gray, 2);
        timeProfile = mean(gray, 1);
        freqAxis = linspace(-0.5, 0.5, size(gray, 1)).';
        timeAxis = linspace(0, 1, size(gray, 2));

        freqMass = sum(freqProfile) + eps;
        timeMass = sum(timeProfile) + eps;
        freqCentroid = sum(freqAxis .* freqProfile) / freqMass;
        timeCentroid = sum(timeAxis .* timeProfile) / timeMass;
        freqSpread = sqrt(sum(((freqAxis - freqCentroid).^2) .* freqProfile) / freqMass);
        timeSpread = sqrt(sum(((timeAxis - timeCentroid).^2) .* timeProfile) / timeMass);

        summary = [
            mean(gray(:)), ...
            std(gray(:)), ...
            mean(gray(:) > 0.5), ...
            freqCentroid, ...
            freqSpread, ...
            timeSpread
        ];

        features(i, :) = single([ ...
            gray(:).', ...
            4 * freqProfile(:).', ...
            4 * timeProfile(:).', ...
            8 * summary ...
        ]);
    end
end

function centroids = build_centroids(trainFeatures, trainLabels, labels)
    centroids = zeros(numel(labels), size(trainFeatures, 2), "single");
    for i = 1:numel(labels)
        idx = trainLabels == labels(i);
        if ~any(idx)
            error("No training samples found for label: %s", labels(i));
        end
        centroids(i, :) = mean(trainFeatures(idx, :), 1);
    end
    centroids = normalize_rows(centroids);
end

function pred = predict_centroid(features, centroids, labels)
    similarity = features * centroids.';
    [~, idx] = max(similarity, [], 2);
    pred = reshape(labels(idx), [], 1);
end

function pred = predict_knn(features, trainFeatures, trainLabels, k)
    similarity = features * trainFeatures.';
    pred = strings(size(features, 1), 1);

    for i = 1:size(features, 1)
        [~, order] = sort(similarity(i, :), "descend");
        nearestLabels = trainLabels(order(1:k));
        uniqueLabels = unique(nearestLabels);
        bestLabel = uniqueLabels(1);
        bestCount = -1;
        bestScore = -inf;

        for j = 1:numel(uniqueLabels)
            label = uniqueLabels(j);
            labelMask = nearestLabels == label;
            labelCount = sum(labelMask);
            labelScore = sum(similarity(i, order(labelMask)));
            if labelCount > bestCount || (labelCount == bestCount && labelScore > bestScore)
                bestLabel = label;
                bestCount = labelCount;
                bestScore = labelScore;
            end
        end

        pred(i) = bestLabel;
    end
end

function normalizedFeatures = normalize_rows(features)
    rowNorms = sqrt(sum(features.^2, 2));
    normalizedFeatures = features ./ max(rowNorms, eps("single"));
end

function gray = normalize_gray(img)
    if ndims(img) == 3
        gray = double(img(:, :, 1)) * 0.2989 + ...
            double(img(:, :, 2)) * 0.5870 + ...
            double(img(:, :, 3)) * 0.1140;
    else
        gray = double(img);
    end

    minVal = min(gray(:));
    maxVal = max(gray(:));
    if maxVal > minVal
        gray = (gray - minVal) / (maxVal - minVal);
    else
        gray = zeros(size(gray));
    end
end

function write_report(path, labels, trainSet, valSet, testSet, methodNames, methodPriority, ...
        valAcc, testAcc, selectedMethod, selectedTestPred)
    fid = fopen(path, "w");
    if fid < 0
        error("Cannot open WTR baseline report: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    log_line(fid, "Mini-WTR Baseline Report");
    log_line(fid, "Train samples: %d", numel(trainSet.ids));
    log_line(fid, "Validation samples: %d", numel(valSet.ids));
    log_line(fid, "Test samples: %d", numel(testSet.ids));
    log_line(fid, "");

    log_line(fid, "Class counts:");
    for i = 1:numel(labels)
        log_line(fid, "  %s: train=%d, val=%d, test=%d", labels(i), ...
            sum(trainSet.labels == labels(i)), ...
            sum(valSet.labels == labels(i)), ...
            sum(testSet.labels == labels(i)));
    end
    log_line(fid, "");

    log_line(fid, "Method accuracy:");
    for i = 1:numel(methodNames)
        log_line(fid, "  %s: val=%.2f%%, test=%.2f%%", ...
            methodNames(i), 100 * valAcc(i), 100 * testAcc(i));
    end
    log_line(fid, "");
    log_line(fid, "Tie-break priority: %s", strjoin(methodPriority, " > "));
    log_line(fid, "Selected method by validation accuracy and fixed tie-break: %s", selectedMethod);
    log_line(fid, "Selected test accuracy: %.2f%%", ...
        100 * mean(selectedTestPred == testSet.labels));
    log_line(fid, "");

    log_line(fid, "Selected test confusion matrix rows=true, columns=predicted:");
    write_confusion_matrix(fid, labels, testSet.labels, selectedTestPred);
end

function write_predictions(path, testSet, testPred, selectedMethod)
    selectedPred = testPred.(char(selectedMethod));
    correct = selectedPred == testSet.labels;
    T = table(testSet.ids, testSet.images, testSet.labels, ...
        testPred.centroid, testPred.knn1, testPred.knn5, selectedPred, correct, ...
        'VariableNames', {'id', 'image', 'true_label', 'pred_centroid', ...
        'pred_knn1', 'pred_knn5', 'selected_pred', 'correct'});
    writetable(T, path);
end

function plot_confusion(path, labels, trueLabels, predLabels, selectedMethod)
    matrixValue = confusion_matrix(labels, trueLabels, predLabels);
    fig = figure("Visible", "off");
    imagesc(matrixValue);
    axis image;
    colorbar;
    colormap(parula);
    set(gca, "XTick", 1:numel(labels), "XTickLabel", labels);
    set(gca, "YTick", 1:numel(labels), "YTickLabel", labels);
    xtickangle(45);
    title("Mini-WTR Baseline Confusion (" + selectedMethod + ")");
    xlabel("Predicted");
    ylabel("True");

    for r = 1:size(matrixValue, 1)
        for c = 1:size(matrixValue, 2)
            text(c, r, num2str(matrixValue(r, c)), ...
                "HorizontalAlignment", "center", "Color", "white", "FontWeight", "bold");
        end
    end

    print(fig, path, "-dpng", "-r150");
    close(fig);
end

function write_confusion_matrix(fid, labels, trueLabels, predLabels)
    matrixValue = confusion_matrix(labels, trueLabels, predLabels);
    log_line(fid, "%14s%14s%14s%14s%14s%14s%14s", "", labels(1), labels(2), labels(3), labels(4), labels(5), labels(6));
    for i = 1:numel(labels)
        log_line(fid, "%14s%14d%14d%14d%14d%14d%14d", labels(i), matrixValue(i, 1), ...
            matrixValue(i, 2), matrixValue(i, 3), matrixValue(i, 4), matrixValue(i, 5), matrixValue(i, 6));
    end
end

function matrixValue = confusion_matrix(labels, trueLabels, predLabels)
    matrixValue = zeros(numel(labels));
    for i = 1:numel(labels)
        for j = 1:numel(labels)
            matrixValue(i, j) = sum(trueLabels == labels(i) & predLabels == labels(j));
        end
    end
end

function value = normalize_path(value)
    value = string(value);
    value = replace(value, "\", filesep);
    value = replace(value, "/", filesep);
end

function value = get_field_as_string(s, fieldName, defaultValue)
    fieldName = char(fieldName);
    value = string(defaultValue);
    if ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    rawValue = s.(fieldName);
    if isempty(rawValue)
        return;
    end
    if isnumeric(rawValue) || islogical(rawValue)
        if isscalar(rawValue)
            value = string(num2str(rawValue));
        else
            value = string(mat2str(rawValue));
        end
    elseif isstring(rawValue)
        value = rawValue;
    elseif ischar(rawValue)
        value = string(rawValue);
    else
        try
            value = string(rawValue);
        catch
            value = string(defaultValue);
        end
    end
end

function log_line(fid, fmt, varargin)
    if nargin == 2
        fprintf("%s\n", fmt);
        fprintf(fid, "%s\n", fmt);
    else
        fprintf(fmt + "\n", varargin{:});
        fprintf(fid, fmt + "\n", varargin{:});
    end
end
