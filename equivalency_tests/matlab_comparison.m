%% Compare MATLAB localizeSources_vIM + source selection with Python output
% Loads registered movie exported from Python zarr, runs the same MATLAB
% pipeline, and reports number of sources + subproblem decomposition.

dir_matlab = getenv('SPINE_MATLAB_REPO');
if isempty(dir_matlab)
    error('SPINE_MATLAB_REPO is not set. Point it at the ophys-slap2-analysis matlab directory.');
end
addpath(genpath(dir_matlab));

%% Load data
data = load(fullfile(scan_dir(), 'registered_ds_act.mat'));
movie_act = double(data.movie_act);
alignHz = data.align_hz;

fprintf('Movie size: %d x %d x %d\n', size(movie_act));
fprintf('align_hz: %.2f\n', alignHz);
fprintf('NaN fraction: %.3f\n', mean(isnan(movie_act), 'all'));

%% Set parameters (matching Python ExtractionConfig)
params.microscope = 'bergamo';
params.sigma_px = 1.33;
params.tau_s = 0.03;
params.dXY = 3;
params.maxSynapseDensity = 0.01;
params.nanThresh = 0.33;
params.baselineWindow_Glu_s = 4.0;
params.denoiseWindow_s = 0.2;
params.alignHz = alignHz;

%% Run localizeSources_vIM
fprintf('\n=== Running localizeSources_vIM ===\n');
tic;
[activityImage, peaks] = localizeSources_vIM(movie_act, [], params, false);
t_loc = toc;
fprintf('Localized %d sources in %.1fs\n', length(peaks.row), t_loc);

% Save activity image for comparison
save(fullfile(scan_dir(), 'matlab_activityImage.mat'), ...
    'activityImage', 'peaks');

%% Source selection (from summarize_LoCo.m lines 226-280)
fprintf('\n=== Running source selection (summarize_LoCo style) ===\n');

actIM = activityImage; % single trial, no averaging needed
medIM = nanmedfilt2(actIM, (2*ceil(1.5*params.dXY)+1).*[1 1]);
actIM = actIM - medIM;

% Iterative NMS
explored = actIM;
pTmp = explored > 0 & explored == ordfilt2(explored, 9, ones(3));
pIM = false(size(actIM));
while any(pTmp(:))
    pIM = pIM | pTmp;
    explored(imdilate(pTmp, ones(5))) = 0;
    pTmp = explored > 0 & explored == ordfilt2(explored, 9, ones(3));
end

% Density threshold
p = actIM(pIM);
sortedP = sort(p, 'descend');
totalPix = sum(~isnan(actIM(:)));
fprintf('Total valid pixels: %d\n', totalPix);
fprintf('Candidate peaks (iterative NMS): %d\n', length(sortedP));

if totalPix > 0 && ~isempty(p)
    threshP = 2 * sortedP(min(end, ceil(totalPix * params.maxSynapseDensity)));
    pp = actIM; pp(~pIM) = 0; pp(pp < threshP) = 0;
    [sources.R, sources.C, sources.V] = find(pp);
    sz = size(pp);
    k = length(sources.R);
else
    k = 0;
end
fprintf('Sources after density threshold: %d\n', k);

%% Create selPix and prune
selPix = false([sz(1:2) k]);
params.selRadius = ceil(2 * params.dXY);
for sourceIx = k:-1:1
    rr = round(sources.R(sourceIx));
    cc = round(sources.C(sourceIx));
    selPix(rr, cc, sourceIx) = true;
    selPix(:,:,sourceIx) = imdilate(selPix(:,:,sourceIx), strel('disk', params.selRadius));
end

% Valid pixel mask
pxAlwaysValid = mean(isnan(movie_act), 3) < params.nanThresh;
selPix = selPix & repmat(pxAlwaysValid, 1, 1, k);

% Prune
keepSources = squeeze(sum(selPix, [1 2])) > 5;
sources.R = sources.R(keepSources);
sources.C = sources.C(keepSources);
sources.V = sources.V(keepSources);
selPix = selPix(:,:,keepSources);
k = sum(keepSources);
fprintf('Sources after pruning: %d\n', k);

%% Subproblem decomposition (from extractTrial.m lines 27-55)
fprintf('\n=== Subproblem decomposition ===\n');

% Match extractTrial.m: create zones and find connected components
params.validKernel = true(3,3);
zones = false(sz(1:2));
for i = 1:k
    zones(round(sources.R(i)), round(sources.C(i))) = true;
end
zoneRadius = ceil(1.5*params.sigma_px + (size(params.validKernel,1)-1)/2);
zones = imdilate(zones, strel('disk', zoneRadius));
CC = bwconncomp(zones, 4);
nProblems = CC.NumObjects;

fprintf('Zone dilation radius: %d\n', zoneRadius);
fprintf('Number of subproblems: %d\n', nProblems);

% Count sources per subproblem
sourcesPerProblem = zeros(1, nProblems);
for problemIx = 1:nProblems
    pxList = CC.PixelIdxList{problemIx};
    selSources = any(sub2ind(sz(1:2), sources.R, sources.C) == pxList', 2);
    sourcesPerProblem(problemIx) = sum(selSources);
end

sourcesPerProblem = sort(sourcesPerProblem, 'descend');
fprintf('Max sources in one subproblem: %d\n', max(sourcesPerProblem));
fprintf('Sources per subproblem: [');
fprintf('%d ', sourcesPerProblem);
fprintf(']\n');

fprintf('\n=== Summary ===\n');
fprintf('Localized peaks: %d\n', length(peaks.row));
fprintf('Selected sources: %d\n', k);
fprintf('Subproblems: %d\n', nProblems);
fprintf('Max per subproblem: %d\n', max(sourcesPerProblem));
