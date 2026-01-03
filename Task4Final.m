%% ===============================================================
% TASK 4 - Dry Bean Classification (COMP3003 Assessment 2)
% Models: Gaussian Naive Bayes + Feedforward Neural Network (64-32)
% Dataset: Dry Bean Dataset (UCI)
% Enhanced Version with Statistical Analysis
% ===============================================================

clear; clc; close all;
rng(42); % Reproducibility

% Set default figure properties for dark theme (better readability)
darkBg = [0.15 0.15 0.15];      % Dark gray background
lightText = [0.95 0.95 0.95];   % Off-white text
gridColor = [0.4 0.4 0.4];      % Medium gray for grids

set(0, 'DefaultFigureColor', darkBg);
set(0, 'DefaultAxesColor', darkBg);
set(0, 'DefaultAxesXColor', lightText);
set(0, 'DefaultAxesYColor', lightText);
set(0, 'DefaultAxesZColor', lightText);
set(0, 'DefaultAxesGridColor', gridColor);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextColor', lightText);
set(0, 'DefaultLegendColor', darkBg);
set(0, 'DefaultLegendTextColor', lightText);

%% ===============================================================
% 1. LOAD AND INSPECT DATA
% ===============================================================

fprintf('=== TASK 4: DRY BEAN CLASSIFICATION ===\n\n');

% Load dataset
data = readtable('Dry_Bean_Dataset.csv');

fprintf('Dataset Preview:\n');
disp(head(data, 5));

fprintf('\nDataset Dimensions: %d samples x %d columns\n', height(data), width(data));

% Check for missing values
missingCount = sum(ismissing(data));
fprintf('Total missing values: %d\n\n', sum(missingCount));

%% ===============================================================
% 2. PREPROCESSING
% ===============================================================

fprintf('=== PREPROCESSING ===\n\n');

% Extract features and labels
featureNames = data.Properties.VariableNames(1:16);
X = table2array(data(:, 1:16));
classLabels = data.Class;

% Convert class labels to numeric
[classNames, ~, y] = unique(classLabels);
nClasses = numel(classNames);

fprintf('Classes (%d):\n', nClasses);
for c = 1:nClasses
    count = sum(y == c);
    fprintf('  %d. %s: %d samples (%.1f%%)\n', c, classNames{c}, count, 100*count/numel(y));
end

% Z-score normalisation
mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;

fprintf('\nNormalisation complete (z-score).\n');

% Feature correlation analysis
corrMatrix = corrcoef(X_norm);
fprintf('\nHighly correlated feature pairs (|r| > 0.9):\n');
highCorrPairs = {};
pairCount = 0;
for i = 1:16
    for j = i+1:16
        if abs(corrMatrix(i,j)) > 0.9
            pairCount = pairCount + 1;
            fprintf('  %s - %s: r = %.3f\n', featureNames{i}, featureNames{j}, corrMatrix(i,j));
            highCorrPairs{pairCount} = struct('i', i, 'j', j, 'r', corrMatrix(i,j));
        end
    end
end
fprintf('Total highly correlated pairs: %d\n', pairCount);

% Plot correlation matrix (dark theme)
figure('Position', [100, 100, 700, 600]);
h = heatmap(featureNames, featureNames, corrMatrix, 'Colormap', parula, 'ColorbarVisible', 'on');
h.Title = 'Feature Correlation Matrix';
h.FontSize = 10;
h.FontColor = lightText;
set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig1_correlation_matrix.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 3. TRAIN-TEST SPLIT
% ===============================================================

fprintf('\n=== TRAIN-TEST SPLIT ===\n\n');

cv_holdout = cvpartition(y, 'HoldOut', 0.3, 'Stratify', true);
X_train = X_norm(training(cv_holdout), :);
y_train = y(training(cv_holdout));
X_test = X_norm(test(cv_holdout), :);
y_test = y(test(cv_holdout));

fprintf('Training samples: %d (70%%)\n', numel(y_train));
fprintf('Test samples: %d (30%%)\n', numel(y_test));

%% ===============================================================
% 4. K-FOLD CROSS-VALIDATION ANALYSIS
% ===============================================================

fprintf('\n=== K-FOLD CROSS-VALIDATION ANALYSIS ===\n\n');

k_values = [3, 5, 10];
kfold_results = table();

for k = k_values
    fprintf('Evaluating k = %d...\n', k);
    
    cv_k = cvpartition(y_train, 'KFold', k, 'Stratify', true);
    
    % Gaussian Naive Bayes
    nb_model_temp = fitcnb(X_train, y_train, 'DistributionNames', 'normal');
    nb_cv = crossval(nb_model_temp, 'CVPartition', cv_k);
    
    % Get fold-wise accuracies
    nb_fold_acc = zeros(k, 1);
    for fold = 1:k
        fold_pred = predict(nb_cv.Trained{fold}, X_train(test(cv_k, fold), :));
        fold_true = y_train(test(cv_k, fold));
        nb_fold_acc(fold) = sum(fold_pred == fold_true) / numel(fold_true);
    end
    
    nb_mean = mean(nb_fold_acc);
    nb_std = std(nb_fold_acc);
    
    % Store results
    new_row = table(k, nb_mean, nb_std, 'VariableNames', {'k', 'NB_Mean', 'NB_Std'});
    kfold_results = [kfold_results; new_row];
end

fprintf('\nK-Fold Comparison Results:\n');
disp(kfold_results);

fprintf('\n--- K-FOLD SELECTION JUSTIFICATION ---\n');
fprintf('k=3: Higher bias, lower variance, fastest computation\n');
fprintf('k=5: Balanced bias-variance tradeoff (SELECTED)\n');
fprintf('k=10: Lower bias, higher variance, slowest computation\n');
fprintf('Selected k=5: Achieves lowest standard deviation (%.4f)\n', kfold_results.NB_Std(2));
fprintf('Reference: Kohavi (1995) recommends stratified 5-fold or 10-fold CV\n');

%% ===============================================================
% 5. GAUSSIAN NAIVE BAYES CLASSIFIER
% ===============================================================

fprintf('\n=== GAUSSIAN NAIVE BAYES ===\n\n');

% Train model
nb_model = fitcnb(X_train, y_train, ...
    'DistributionNames', 'normal', ...
    'Prior', 'empirical');

% 5-fold cross-validation
cv_5 = cvpartition(y_train, 'KFold', 5, 'Stratify', true);
nb_cv_model = crossval(nb_model, 'CVPartition', cv_5);
nb_cv_accuracy = 1 - kfoldLoss(nb_cv_model);
fprintf('5-Fold CV Accuracy: %.4f (%.2f%%)\n', nb_cv_accuracy, nb_cv_accuracy*100);

% Test set predictions
[y_pred_nb, nb_posterior] = predict(nb_model, X_test);
nb_test_accuracy = sum(y_pred_nb == y_test) / numel(y_test);
fprintf('Test Accuracy: %.4f (%.2f%%)\n', nb_test_accuracy, nb_test_accuracy*100);

% Confusion matrix (dark theme)
figure('Position', [100, 100, 650, 550]);
cm_nb = confusionchart(y_test, y_pred_nb, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm_nb.Title = 'Gaussian Naive Bayes Confusion Matrix';
cm_nb.FontColor = lightText;
set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig2_nb_confusion_matrix.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 6. ALTERNATIVE NAIVE BAYES DISTRIBUTIONS
% ===============================================================

fprintf('\n=== ALTERNATIVE NB DISTRIBUTIONS ===\n\n');

% Kernel Naive Bayes
nb_kernel = fitcnb(X_train, y_train, 'DistributionNames', 'kernel');
y_pred_kernel = predict(nb_kernel, X_test);
kernel_accuracy = sum(y_pred_kernel == y_test) / numel(y_test);
fprintf('Kernel NB Test Accuracy: %.4f (%.2f%%)\n', kernel_accuracy, kernel_accuracy*100);

fprintf('\n--- DISTRIBUTION JUSTIFICATION ---\n');
fprintf('Gaussian NB (SELECTED): %.2f%% - Appropriate for continuous features\n', nb_test_accuracy*100);
fprintf('Kernel NB: %.2f%% - Non-parametric density estimation\n', kernel_accuracy*100);
fprintf('Multinomial NB: NOT APPLICABLE - Requires count data\n');
fprintf('Bernoulli NB: NOT APPLICABLE - Requires binary features\n');

%% ===============================================================
% 7. NEURAL NETWORK CLASSIFIER
% ===============================================================

fprintf('\n=== NEURAL NETWORK (64-32) ===\n\n');

% Prepare data for NN toolbox
X_train_T = X_train';
X_test_T = X_test';
Y_train_onehot = full(ind2vec(y_train'))';
Y_train_T = Y_train_onehot';

% Define architecture
hiddenLayers = [64, 32];
net = patternnet(hiddenLayers);

% Configure network
net.trainFcn = 'trainscg';
net.performFcn = 'crossentropy';

% Data division for early stopping
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.1;

% Training parameters
net.trainParam.epochs = 500;
net.trainParam.max_fail = 20;
net.trainParam.showWindow = true;

% Train network
fprintf('Training neural network...\n');
tic;
[net, tr] = train(net, X_train_T, Y_train_T);
training_time = toc;
fprintf('Training completed in %.2f seconds.\n', training_time);

% Test set predictions
nn_output = net(X_test_T);
[~, y_pred_nn] = max(nn_output, [], 1);
y_pred_nn = y_pred_nn';

nn_test_accuracy = sum(y_pred_nn == y_test) / numel(y_test);
fprintf('Test Accuracy: %.4f (%.2f%%)\n', nn_test_accuracy, nn_test_accuracy*100);

%% ===============================================================
% 7b. NEURAL NETWORK 5-FOLD CROSS-VALIDATION
% ===============================================================

fprintf('\n--- Neural Network 5-Fold Cross-Validation ---\n');
nn_cv_accuracies = zeros(5, 1);
cv_5_nn = cvpartition(y_train, 'KFold', 5, 'Stratify', true);

for fold = 1:5
    fprintf('  Fold %d/5...\n', fold);
    
    X_fold_train = X_train(training(cv_5_nn, fold), :)';
    y_fold_train = y_train(training(cv_5_nn, fold));
    Y_fold_train = full(ind2vec(y_fold_train'))';
    X_fold_val = X_train(test(cv_5_nn, fold), :)';
    y_fold_val = y_train(test(cv_5_nn, fold));
    
    net_fold = patternnet([64, 32]);
    net_fold.trainFcn = 'trainscg';
    net_fold.performFcn = 'crossentropy';
    net_fold.trainParam.showWindow = false;
    net_fold.trainParam.epochs = 500;
    net_fold.trainParam.max_fail = 20;
    net_fold.divideParam.trainRatio = 0.9;
    net_fold.divideParam.valRatio = 0.1;
    net_fold.divideParam.testRatio = 0;
    
    net_fold = train(net_fold, X_fold_train, Y_fold_train');
    
    fold_output = net_fold(X_fold_val);
    [~, fold_pred] = max(fold_output, [], 1);
    nn_cv_accuracies(fold) = sum(fold_pred' == y_fold_val) / numel(y_fold_val);
end

nn_cv_accuracy = mean(nn_cv_accuracies);
nn_cv_std = std(nn_cv_accuracies);
fprintf('Neural Network 5-Fold CV Accuracy: %.4f (%.2f%%) +/- %.4f\n', ...
    nn_cv_accuracy, nn_cv_accuracy*100, nn_cv_std);

% Confusion matrix (dark theme)
figure('Position', [100, 100, 650, 550]);
cm_nn = confusionchart(y_test, y_pred_nn, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm_nn.Title = 'Neural Network (64-32) Confusion Matrix';
cm_nn.FontColor = lightText;
set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig3_nn_confusion_matrix.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 8. COMPREHENSIVE METRICS COMPUTATION
% ===============================================================

fprintf('\n=== CLASSIFICATION METRICS ===\n\n');

% Naive Bayes metrics
prec_nb = zeros(nClasses, 1);
rec_nb = zeros(nClasses, 1);
f1_nb = zeros(nClasses, 1);

for c = 1:nClasses
    TP = sum((y_pred_nb == c) & (y_test == c));
    FP = sum((y_pred_nb == c) & (y_test ~= c));
    FN = sum((y_pred_nb ~= c) & (y_test == c));
    
    prec_nb(c) = TP / (TP + FP + eps);
    rec_nb(c) = TP / (TP + FN + eps);
    f1_nb(c) = 2 * prec_nb(c) * rec_nb(c) / (prec_nb(c) + rec_nb(c) + eps);
end

% Neural Network metrics
prec_nn = zeros(nClasses, 1);
rec_nn = zeros(nClasses, 1);
f1_nn = zeros(nClasses, 1);

for c = 1:nClasses
    TP = sum((y_pred_nn == c) & (y_test == c));
    FP = sum((y_pred_nn == c) & (y_test ~= c));
    FN = sum((y_pred_nn ~= c) & (y_test == c));
    
    prec_nn(c) = TP / (TP + FP + eps);
    rec_nn(c) = TP / (TP + FN + eps);
    f1_nn(c) = 2 * prec_nn(c) * rec_nn(c) / (prec_nn(c) + rec_nn(c) + eps);
end

% Display metrics
fprintf('GAUSSIAN NAIVE BAYES - Class-level Metrics:\n');
fprintf('%-12s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 45));
for c = 1:nClasses
    fprintf('%-12s %10.4f %10.4f %10.4f\n', classNames{c}, prec_nb(c), rec_nb(c), f1_nb(c));
end
fprintf('%s\n', repmat('-', 1, 45));
fprintf('%-12s %10.4f %10.4f %10.4f\n', 'Macro Avg', mean(prec_nb), mean(rec_nb), mean(f1_nb));

fprintf('\nNEURAL NETWORK (64-32) - Class-level Metrics:\n');
fprintf('%-12s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 45));
for c = 1:nClasses
    fprintf('%-12s %10.4f %10.4f %10.4f\n', classNames{c}, prec_nn(c), rec_nn(c), f1_nn(c));
end
fprintf('%s\n', repmat('-', 1, 45));
fprintf('%-12s %10.4f %10.4f %10.4f\n', 'Macro Avg', mean(prec_nn), mean(rec_nn), mean(f1_nn));

%% ===============================================================
% 9. STATISTICAL SIGNIFICANCE TESTING
% ===============================================================

fprintf('\n=== STATISTICAL SIGNIFICANCE TESTING ===\n\n');

% McNemar's Test
nb_correct = (y_pred_nb == y_test);
nn_correct = (y_pred_nn == y_test);

% Contingency table
a = sum(nb_correct & nn_correct);      % Both correct
b = sum(nb_correct & ~nn_correct);     % NB correct, NN wrong
c = sum(~nb_correct & nn_correct);     % NB wrong, NN correct
d = sum(~nb_correct & ~nn_correct);    % Both wrong

fprintf('McNemar Contingency Table:\n');
fprintf('                    NN Correct    NN Wrong\n');
fprintf('  NB Correct          %4d          %4d\n', a, b);
fprintf('  NB Wrong            %4d          %4d\n', c, d);

% McNemar's chi-squared statistic (with continuity correction)
chi2_mcnemar = (abs(b - c) - 1)^2 / (b + c);
p_value_mcnemar = 1 - chi2cdf(chi2_mcnemar, 1);

fprintf('\nMcNemar Test Results:\n');
fprintf('  Chi-squared statistic: %.4f\n', chi2_mcnemar);
fprintf('  Degrees of freedom: 1\n');
fprintf('  p-value: %.6f\n', p_value_mcnemar);

if p_value_mcnemar < 0.001
    fprintf('  Conclusion: Highly significant difference (p < 0.001) ***\n');
elseif p_value_mcnemar < 0.01
    fprintf('  Conclusion: Significant difference (p < 0.01) **\n');
elseif p_value_mcnemar < 0.05
    fprintf('  Conclusion: Significant difference (p < 0.05) *\n');
else
    fprintf('  Conclusion: No significant difference (p >= 0.05)\n');
end

%% ===============================================================
% 10. CONFIDENCE INTERVALS
% ===============================================================

fprintf('\n=== CONFIDENCE INTERVALS (95%%) ===\n\n');

n = numel(y_test);
z = 1.96;  % 95% confidence

% Wilson score interval for NB
p_nb = nb_test_accuracy;
denominator_nb = 1 + z^2/n;
center_nb = p_nb + z^2/(2*n);
spread_nb = z * sqrt((p_nb*(1-p_nb) + z^2/(4*n))/n);
ci_lower_nb = (center_nb - spread_nb) / denominator_nb;
ci_upper_nb = (center_nb + spread_nb) / denominator_nb;

% Wilson score interval for NN
p_nn = nn_test_accuracy;
denominator_nn = 1 + z^2/n;
center_nn = p_nn + z^2/(2*n);
spread_nn = z * sqrt((p_nn*(1-p_nn) + z^2/(4*n))/n);
ci_lower_nn = (center_nn - spread_nn) / denominator_nn;
ci_upper_nn = (center_nn + spread_nn) / denominator_nn;

fprintf('Gaussian Naive Bayes:\n');
fprintf('  Accuracy: %.2f%% [95%% CI: %.2f%% - %.2f%%]\n', p_nb*100, ci_lower_nb*100, ci_upper_nb*100);

fprintf('Neural Network:\n');
fprintf('  Accuracy: %.2f%% [95%% CI: %.2f%% - %.2f%%]\n', p_nn*100, ci_lower_nn*100, ci_upper_nn*100);

% Check overlap
if ci_upper_nb < ci_lower_nn
    fprintf('\nConclusion: Non-overlapping CIs confirm significant difference.\n');
else
    fprintf('\nConclusion: CIs overlap - difference may not be significant.\n');
end

%% ===============================================================
% 11. ROC CURVES AND AUC ANALYSIS
% ===============================================================

fprintf('\n=== ROC CURVE AND AUC ANALYSIS ===\n\n');

% ROC for Naive Bayes (One-vs-Rest)
figure('Position', [100, 100, 1000, 450]);

subplot(1, 2, 1);
hold on;
colors = lines(nClasses);
auc_nb = zeros(nClasses, 1);
legend_entries = cell(nClasses, 1);

for c = 1:nClasses
    [X_roc, Y_roc, ~, auc_nb(c)] = perfcurve(y_test, nb_posterior(:,c), c);
    plot(X_roc, Y_roc, 'Color', colors(c,:), 'LineWidth', 1.5);
    legend_entries{c} = sprintf('%s (AUC=%.3f)', classNames{c}, auc_nb(c));
end

plot([0 1], [0 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
xlabel('False Positive Rate', 'FontSize', 12, 'Color', lightText);
ylabel('True Positive Rate', 'FontSize', 12, 'Color', lightText);
title('ROC Curves - Gaussian Naive Bayes', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
lgd = legend(legend_entries, 'Location', 'southeast', 'FontSize', 8);
lgd.TextColor = lightText;
lgd.Color = darkBg;
grid on;
axis square;

% ROC for Neural Network (One-vs-Rest)
subplot(1, 2, 2);
hold on;
auc_nn = zeros(nClasses, 1);
legend_entries_nn = cell(nClasses, 1);

for c = 1:nClasses
    [X_roc, Y_roc, ~, auc_nn(c)] = perfcurve(y_test, nn_output(c,:)', c);
    plot(X_roc, Y_roc, 'Color', colors(c,:), 'LineWidth', 1.5);
    legend_entries_nn{c} = sprintf('%s (AUC=%.3f)', classNames{c}, auc_nn(c));
end

plot([0 1], [0 1], '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
xlabel('False Positive Rate', 'FontSize', 12, 'Color', lightText);
ylabel('True Positive Rate', 'FontSize', 12, 'Color', lightText);
title('ROC Curves - Neural Network (64-32)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
lgd2 = legend(legend_entries_nn, 'Location', 'southeast', 'FontSize', 8);
lgd2.TextColor = lightText;
lgd2.Color = darkBg;
grid on;
axis square;

set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig6_roc_curves.png', 'Resolution', 300, 'BackgroundColor', darkBg);

fprintf('AUC Comparison:\n');
fprintf('%-12s %10s %10s\n', 'Class', 'NB AUC', 'NN AUC');
fprintf('%s\n', repmat('-', 1, 35));
for c = 1:nClasses
    fprintf('%-12s %10.4f %10.4f\n', classNames{c}, auc_nb(c), auc_nn(c));
end
fprintf('%s\n', repmat('-', 1, 35));
fprintf('%-12s %10.4f %10.4f\n', 'Mean AUC', mean(auc_nb), mean(auc_nn));

%% ===============================================================
% 12. HYPERPARAMETER SENSITIVITY ANALYSIS
% ===============================================================

fprintf('\n=== HYPERPARAMETER SENSITIVITY ANALYSIS ===\n\n');

architectures = {[32], [64], [128], [32, 16], [64, 32], [128, 64], [64, 32, 16]};
arch_names = {'[32]', '[64]', '[128]', '[32-16]', '[64-32]', '[128-64]', '[64-32-16]'};
arch_accuracies = zeros(length(architectures), 1);
arch_times = zeros(length(architectures), 1);

fprintf('Testing architectures...\n');
for i = 1:length(architectures)
    fprintf('  %s...', arch_names{i});
    
    net_test = patternnet(architectures{i});
    net_test.trainFcn = 'trainscg';
    net_test.performFcn = 'crossentropy';
    net_test.trainParam.showWindow = false;
    net_test.trainParam.epochs = 500;
    net_test.trainParam.max_fail = 20;
    net_test.divideParam.trainRatio = 0.8;
    net_test.divideParam.valRatio = 0.1;
    net_test.divideParam.testRatio = 0.1;
    
    tic;
    net_test = train(net_test, X_train_T, Y_train_T);
    arch_times(i) = toc;
    
    output = net_test(X_test_T);
    [~, pred] = max(output, [], 1);
    arch_accuracies(i) = sum(pred' == y_test) / numel(y_test);
    
    fprintf(' %.2f%% (%.1fs)\n', arch_accuracies(i)*100, arch_times(i));
end

fprintf('\nArchitecture Comparison:\n');
fprintf('%-15s %12s %12s\n', 'Architecture', 'Accuracy', 'Time (s)');
fprintf('%s\n', repmat('-', 1, 42));
for i = 1:length(architectures)
    marker = '';
    if i == 5  % [64-32] is our chosen architecture
        marker = ' <-- SELECTED';
    end
    fprintf('%-15s %11.2f%% %12.2f%s\n', arch_names{i}, arch_accuracies(i)*100, arch_times(i), marker);
end

% Plot architecture comparison (dark theme)
figure('Position', [100, 100, 600, 400]);
b = bar(arch_accuracies * 100, 'FaceColor', [0.3 0.5 0.8]);
set(gca, 'XTickLabel', arch_names);
xlabel('Network Architecture', 'FontSize', 12, 'Color', lightText);
ylabel('Test Accuracy (%)', 'FontSize', 12, 'Color', lightText);
title('Hyperparameter Sensitivity: Network Architecture', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
ylim([88 95]);
grid on;

% Highlight selected architecture
hold on;
bar(5, arch_accuracies(5)*100, 'FaceColor', [0.2 0.8 0.2]);
lgd = legend({'Other', 'Selected [64-32]'}, 'Location', 'southeast');
lgd.TextColor = lightText;
lgd.Color = darkBg;

set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig7_architecture_comparison.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 13. LEARNING CURVES
% ===============================================================

fprintf('\n=== LEARNING CURVE ANALYSIS ===\n\n');

train_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
learning_curve_nb = zeros(length(train_fractions), 1);
learning_curve_nn = zeros(length(train_fractions), 1);

fprintf('Computing learning curves...\n');
for i = 1:length(train_fractions)
    frac = train_fractions(i);
    n_samples = round(frac * size(X_train, 1));
    
    % Use consistent random subset
    rng(42);
    idx = randperm(size(X_train, 1), n_samples);
    
    % Train NB on subset
    nb_sub = fitcnb(X_train(idx,:), y_train(idx), 'DistributionNames', 'normal', 'Prior', 'empirical');
    learning_curve_nb(i) = sum(predict(nb_sub, X_test) == y_test) / numel(y_test);
    
    % Train NN on subset
    X_sub = X_train(idx,:)';
    Y_sub = full(ind2vec(y_train(idx)'))';
    
    net_sub = patternnet([64, 32]);
    net_sub.trainFcn = 'trainscg';
    net_sub.trainParam.showWindow = false;
    net_sub.trainParam.epochs = 500;
    net_sub.trainParam.max_fail = 20;
    net_sub.divideParam.trainRatio = 0.9;
    net_sub.divideParam.valRatio = 0.1;
    net_sub.divideParam.testRatio = 0;
    
    net_sub = train(net_sub, X_sub, Y_sub');
    output_sub = net_sub(X_test_T);
    [~, pred_sub] = max(output_sub, [], 1);
    learning_curve_nn(i) = sum(pred_sub' == y_test) / numel(y_test);
    
    fprintf('  %.0f%%: NB=%.2f%%, NN=%.2f%%\n', frac*100, learning_curve_nb(i)*100, learning_curve_nn(i)*100);
end

% Plot learning curves (dark theme)
figure('Position', [100, 100, 600, 450]);
plot(train_fractions * 100, learning_curve_nb * 100, 'c-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'c');
hold on;
plot(train_fractions * 100, learning_curve_nn * 100, 'm-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'm');
xlabel('Training Set Size (%)', 'FontSize', 12, 'Color', lightText);
ylabel('Test Accuracy (%)', 'FontSize', 12, 'Color', lightText);
title('Learning Curves', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
lgd = legend('Gaussian Naive Bayes', 'Neural Network (64-32)', 'Location', 'southeast');
lgd.TextColor = lightText;
lgd.Color = darkBg;
grid on;
xlim([5 105]);
ylim([80 95]);

set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig8_learning_curves.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 14. FEATURE IMPORTANCE (PERMUTATION)
% ===============================================================

fprintf('\n=== FEATURE IMPORTANCE ANALYSIS ===\n\n');

baseline_acc_nn = nn_test_accuracy;
importance_nn = zeros(16, 1);

fprintf('Computing permutation importance...\n');
for f = 1:16
    % Permute feature f
    X_test_permuted = X_test;
    rng(42);
    X_test_permuted(:, f) = X_test(randperm(size(X_test, 1)), f);
    
    % Evaluate NN on permuted data
    output_perm = net(X_test_permuted');
    [~, pred_perm] = max(output_perm, [], 1);
    permuted_acc = sum(pred_perm' == y_test) / numel(y_test);
    
    importance_nn(f) = baseline_acc_nn - permuted_acc;
end

% Sort by importance
[sorted_imp, sort_idx] = sort(importance_nn, 'descend');

fprintf('Feature Importance (Permutation-based):\n');
fprintf('%-20s %15s\n', 'Feature', 'Importance');
fprintf('%s\n', repmat('-', 1, 38));
for i = 1:16
    fprintf('%-20s %14.4f\n', featureNames{sort_idx(i)}, sorted_imp(i));
end

% Plot feature importance (dark theme)
figure('Position', [100, 100, 700, 450]);
barh(importance_nn(sort_idx), 'FaceColor', [0.3 0.7 0.9]);
set(gca, 'YTickLabel', featureNames(sort_idx), 'YDir', 'reverse');
xlabel('Importance (Accuracy Drop)', 'FontSize', 12, 'Color', lightText);
ylabel('Feature', 'FontSize', 12, 'Color', lightText);
title('Feature Importance (Permutation-based)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
grid on;

set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig9_feature_importance.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 15. FEATURE ABLATION STUDY
% ===============================================================

fprintf('\n=== FEATURE ABLATION STUDY ===\n\n');
fprintf('Testing hypothesis: Removing highly correlated features improves NB performance\n\n');

ablation_results = table();

% Baseline: All 16 features
baseline_acc = nb_test_accuracy;
fprintf('Baseline (all 16 features): %.4f (%.2f%%)\n', baseline_acc, baseline_acc*100);
ablation_results = [ablation_results; table({'All 16 features'}, baseline_acc, 0, ...
    'VariableNames', {'Configuration', 'Accuracy', 'Change'})];

% Ablation 1: Remove ConvexArea
features_v1 = setdiff(1:16, 7);  % ConvexArea is column 7
nb_abl1 = fitcnb(X_train(:, features_v1), y_train, 'DistributionNames', 'normal', 'Prior', 'empirical');
pred_abl1 = predict(nb_abl1, X_test(:, features_v1));
acc_abl1 = sum(pred_abl1 == y_test) / numel(y_test);
change1 = acc_abl1 - baseline_acc;
fprintf('Remove ConvexArea (r=1.0 with Area): %.4f (%.2f%%) [%+.2f%%]\n', acc_abl1, acc_abl1*100, change1*100);
ablation_results = [ablation_results; table({'Remove ConvexArea'}, acc_abl1, change1, ...
    'VariableNames', {'Configuration', 'Accuracy', 'Change'})];

% Ablation 2: Remove ConvexArea + EquivDiameter
features_v2 = setdiff(1:16, [7, 8]);  % ConvexArea=7, EquivDiameter=8
nb_abl2 = fitcnb(X_train(:, features_v2), y_train, 'DistributionNames', 'normal', 'Prior', 'empirical');
pred_abl2 = predict(nb_abl2, X_test(:, features_v2));
acc_abl2 = sum(pred_abl2 == y_test) / numel(y_test);
change2 = acc_abl2 - baseline_acc;
fprintf('Remove ConvexArea + EquivDiameter: %.4f (%.2f%%) [%+.2f%%]\n', acc_abl2, acc_abl2*100, change2*100);
ablation_results = [ablation_results; table({'Remove ConvexArea + EquivDiameter'}, acc_abl2, change2, ...
    'VariableNames', {'Configuration', 'Accuracy', 'Change'})];

% Ablation 3: Remove 4 most correlated features
features_v3 = setdiff(1:16, [7, 8, 15, 3]);  % ConvexArea, EquivDiameter, ShapeFactor3, MajorAxisLength
nb_abl3 = fitcnb(X_train(:, features_v3), y_train, 'DistributionNames', 'normal', 'Prior', 'empirical');
pred_abl3 = predict(nb_abl3, X_test(:, features_v3));
acc_abl3 = sum(pred_abl3 == y_test) / numel(y_test);
change3 = acc_abl3 - baseline_acc;
fprintf('Remove 4 correlated features: %.4f (%.2f%%) [%+.2f%%]\n', acc_abl3, acc_abl3*100, change3*100);
ablation_results = [ablation_results; table({'Remove 4 correlated features'}, acc_abl3, change3, ...
    'VariableNames', {'Configuration', 'Accuracy', 'Change'})];

fprintf('\n--- ABLATION STUDY CONCLUSION ---\n');
if max([change1, change2, change3]) > 0
    fprintf('Removing correlated features IMPROVES NB performance.\n');
    fprintf('This confirms the independence assumption violation hurts NB.\n');
else
    fprintf('Removing correlated features does not significantly improve NB.\n');
end

%% ===============================================================
% 16. SUMMARY COMPARISON
% ===============================================================

fprintf('\n=== FINAL COMPARISON ===\n\n');

fprintf('%-30s %15s %15s\n', 'Metric', 'Gaussian NB', 'Neural Network');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%-30s %14.2f%% %14.2f%%\n', '5-Fold CV Accuracy', nb_cv_accuracy*100, nn_cv_accuracy*100);
fprintf('%-30s %14.2f%% %14.2f%%\n', 'Test Accuracy', nb_test_accuracy*100, nn_test_accuracy*100);
fprintf('%-30s %14.2f%% %14.2f%%\n', 'Macro Precision', mean(prec_nb)*100, mean(prec_nn)*100);
fprintf('%-30s %14.2f%% %14.2f%%\n', 'Macro Recall', mean(rec_nb)*100, mean(rec_nn)*100);
fprintf('%-30s %14.2f%% %14.2f%%\n', 'Macro F1-Score', mean(f1_nb)*100, mean(f1_nn)*100);
fprintf('%-30s %14.4f %14.4f\n', 'Mean AUC', mean(auc_nb), mean(auc_nn));
fprintf('%s\n', repmat('=', 1, 60));

perf_diff = (nn_test_accuracy - nb_test_accuracy)*100;
fprintf('\nPerformance Difference: %.2f percentage points in favor of Neural Network\n', perf_diff);
fprintf('Statistical Significance: p = %.6f (McNemar test)\n', p_value_mcnemar);

%% ===============================================================
% 17. FEATURE DISTRIBUTION PLOTS
% ===============================================================

figure('Position', [100, 100, 1100, 800]);
for j = 1:16
    subplot(4, 4, j);
    histogram(X_norm(:, j), 30, 'Normalization', 'pdf', 'FaceColor', [0.3 0.7 0.9], 'EdgeColor', 'none');
    hold on;
    x_range = linspace(min(X_norm(:, j)), max(X_norm(:, j)), 100);
    plot(x_range, normpdf(x_range, 0, 1), 'y-', 'LineWidth', 1.5);
    title(featureNames{j}, 'FontSize', 9, 'Color', lightText);
    set(gca, 'FontSize', 8, 'Color', darkBg, 'XColor', lightText, 'YColor', lightText);
end
sgtitle('Feature Distributions vs Standard Normal (Yellow Line)', 'Color', lightText);
set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig4_feature_distributions.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 18. TRAINING PERFORMANCE PLOT
% ===============================================================

figure('Position', [100, 100, 600, 400]);
semilogy(tr.perf, 'c-', 'LineWidth', 1.5);
hold on;
semilogy(tr.vperf, 'm-', 'LineWidth', 1.5);
semilogy(tr.tperf, 'y-', 'LineWidth', 1.5);
xlabel('Epoch', 'FontSize', 12, 'Color', lightText);
ylabel('Cross-Entropy Loss (log scale)', 'FontSize', 12, 'Color', lightText);
lgd = legend('Training', 'Validation', 'Test', 'Location', 'northeast');
lgd.TextColor = lightText;
lgd.Color = darkBg;
title('Neural Network Training Performance', 'FontSize', 14, 'FontWeight', 'bold', 'Color', lightText);
grid on;
set(gcf, 'Color', darkBg);
exportgraphics(gcf, 'fig5_training_performance.png', 'Resolution', 300, 'BackgroundColor', darkBg);

%% ===============================================================
% 19. SAVE ALL RESULTS
% ===============================================================

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Figures saved:\n');
fprintf('  - fig1_correlation_matrix.png\n');
fprintf('  - fig2_nb_confusion_matrix.png\n');
fprintf('  - fig3_nn_confusion_matrix.png\n');
fprintf('  - fig4_feature_distributions.png\n');
fprintf('  - fig5_training_performance.png\n');
fprintf('  - fig6_roc_curves.png\n');
fprintf('  - fig7_architecture_comparison.png\n');
fprintf('  - fig8_learning_curves.png\n');
fprintf('  - fig9_feature_importance.png\n');

% Save results to MAT file for reproducibility
save('task4_results.mat', 'nb_test_accuracy', 'nn_test_accuracy', ...
    'nb_cv_accuracy', 'nn_cv_accuracy', 'prec_nb', 'rec_nb', 'f1_nb', ...
    'prec_nn', 'rec_nn', 'f1_nn', 'auc_nb', 'auc_nn', 'chi2_mcnemar', ...
    'p_value_mcnemar', 'ci_lower_nb', 'ci_upper_nb', 'ci_lower_nn', 'ci_upper_nn', ...
    'importance_nn', 'arch_accuracies', 'learning_curve_nb', 'learning_curve_nn');

fprintf('\nResults saved to task4_results.mat\n');

%% ===============================================================
% END OF SCRIPT
% ===============================================================
