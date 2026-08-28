%% Step-by-step localizeSources_vIM with intermediate saves
% Runs each processing step and saves the result for Python comparison.

dir_matlab = getenv('SPINE_MATLAB_REPO');
if isempty(dir_matlab)
    error('SPINE_MATLAB_REPO is not set. Point it at the ophys-slap2-analysis matlab directory.');
end
addpath(genpath(dir_matlab));

%% Load data
data = load(fullfile(scan_dir(), 'registered_ds_act.mat'));
IM = double(data.movie_act);
alignHz = data.align_hz;
outdir = scan_dir();

%% Parameters
tau = 0.03 * alignHz;
sigma = 1.33;
baselineWindow = ceil(4.0 * alignHz);
denoiseWindow = ceil(0.2 * alignHz);
nanThresh = 0.33;

fprintf('tau=%.2f frames, sigma=%.2f, baselineWindow=%d, denoiseWindow=%d\n', ...
    tau, sigma, baselineWindow, denoiseWindow);

%% Step 0: Initial NaN handling
nans = isnan(IM);
sz = size(IM);
valid = mean(nans, 3) < nanThresh;
IMf = IM; clear IM;
IMf(repmat(~valid, 1, 1, size(IMf,3))) = nan;
nans = isnan(IMf);

% Save a single pixel's time series for debugging
% Pick a pixel that has some NaN (border pixel)
nanFrac = mean(nans, 3);
borderPx = find(nanFrac > 0.05 & nanFrac < 0.3 & valid);
interiorPx = find(nanFrac == 0 & valid);

% Pick specific pixels for comparison
if ~isempty(borderPx)
    [br, bc] = ind2sub(sz(1:2), borderPx(1));
    fprintf('Border pixel: (%d, %d), NaN frac: %.3f\n', br, bc, nanFrac(br,bc));
else
    br = 35; bc = 75;
end
if ~isempty(interiorPx)
    [ir, ic] = ind2sub(sz(1:2), interiorPx(round(end/2)));
    fprintf('Interior pixel: (%d, %d)\n', ir, ic);
else
    ir = 35; ic = 75;
end

%% Step 1: Smoothing (movmean)
IMfden = smoothdata(IMf, 3, 'movmean', denoiseWindow, 'omitnan');

% Save pixel traces
step1_border_raw = squeeze(IMf(br, bc, :));
step1_border_smooth = squeeze(IMfden(br, bc, :));
step1_interior_raw = squeeze(IMf(ir, ic, :));
step1_interior_smooth = squeeze(IMfden(ir, ic, :));

% Save a single frame for spatial comparison
frame_idx = 1000;
step1_frame_raw = IMf(:,:,frame_idx);
step1_frame_smooth = IMfden(:,:,frame_idx);

save(fullfile(outdir, 'matlab_step1_smooth.mat'), ...
    'step1_border_raw', 'step1_border_smooth', ...
    'step1_interior_raw', 'step1_interior_smooth', ...
    'step1_frame_raw', 'step1_frame_smooth', ...
    'br', 'bc', 'ir', 'ic', 'denoiseWindow');

% Save multiple frames for better statistics
step1_frames_smooth = IMfden(:,:,[100 500 1000 5000 10000]);
save(fullfile(outdir, 'matlab_step1b_frames.mat'), 'step1_frames_smooth');

%% Step 2: Baseline (movmedian on smoothed)
IMb = smoothdata(IMfden, 3, 'movmedian', baselineWindow, 'omitnan');
IMf_hp = IMf - IMb;  % raw - smoothed baseline

step2_border_baseline = squeeze(IMb(br, bc, :));
step2_border_hp = squeeze(IMf_hp(br, bc, :));
step2_interior_baseline = squeeze(IMb(ir, ic, :));
step2_interior_hp = squeeze(IMf_hp(ir, ic, :));

step2_frame_baseline = IMb(:,:,frame_idx);
step2_frame_hp = IMf_hp(:,:,frame_idx);
step2_frames_baseline = IMb(:,:,[100 500 1000 5000 10000]);
step2_frames_hp = IMf_hp(:,:,[100 500 1000 5000 10000]);
save(fullfile(outdir, 'matlab_step2_baseline.mat'), ...
    'step2_border_baseline', 'step2_border_hp', ...
    'step2_interior_baseline', 'step2_interior_hp', ...
    'step2_frame_baseline', 'step2_frame_hp', ...
    'step2_frames_baseline', 'step2_frames_hp');

%% Step 3: MAD noise
stdIM = movmad(IMfden - IMb, baselineWindow, 3, 'omitmissing') ./ 0.6741891400433162 .* denoiseWindow;

step3_border_std = squeeze(stdIM(br, bc, :));
step3_interior_std = squeeze(stdIM(ir, ic, :));

% Z-score
IMf_z = IMf_hp ./ stdIM;
step3_border_z = squeeze(IMf_z(br, bc, :));
step3_interior_z = squeeze(IMf_z(ir, ic, :));

step3_frame_std = stdIM(:,:,frame_idx);
step3_frame_z = IMf_z(:,:,frame_idx);
step3_frames_z = IMf_z(:,:,[100 500 1000 5000 10000]);
save(fullfile(outdir, 'matlab_step3_noise.mat'), ...
    'step3_border_std', 'step3_interior_std', ...
    'step3_border_z', 'step3_interior_z', ...
    'step3_frame_std', 'step3_frame_z', 'step3_frames_z');

clear IMfden IMb IMf_hp stdIM;

%% Step 4: Matched filter
gamma = exp(-1/tau);
mem = max(0, gamma * IMf_z(:,:,end));
for t = size(IMf_z,3):-1:1
    IMt = IMf_z(:,:,t);
    nanst = isnan(IMt);
    IMt(nanst) = mem(nanst);
    IMf_z(:,:,t) = gamma*mem + (1-gamma)*IMt;
    mem = IMf_z(:,:,t);
end
IMf_z(nans) = nan;

step4_border_mf = squeeze(IMf_z(br, bc, :));
step4_interior_mf = squeeze(IMf_z(ir, ic, :));
step4_frame = IMf_z(:,:,frame_idx);

step4_frames = IMf_z(:,:,[100 500 1000 5000 10000]);
save(fullfile(outdir, 'matlab_step4_matchedfilter.mat'), ...
    'step4_border_mf', 'step4_interior_mf', 'step4_frame', 'step4_frames');

%% Step 5: DoG
IMf_z(nans) = 0;
IMf_dog = imgaussfilt(IMf_z, [sigma sigma]);
IMf_dog = IMf_dog - imgaussfilt(IMf_dog, 5*[sigma sigma]);
IMf_dog(nans) = nan;

step5_frame = IMf_dog(:,:,frame_idx);
step5_border_dog = squeeze(IMf_dog(br, bc, :));
step5_interior_dog = squeeze(IMf_dog(ir, ic, :));

step5_frames = IMf_dog(:,:,[100 500 1000 5000 10000]);
save(fullfile(outdir, 'matlab_step5_dog.mat'), ...
    'step5_frame', 'step5_border_dog', 'step5_interior_dog', 'step5_frames');

%% Step 6: NMS + activity image
skIm = zeros(sz(1:2));
for fr = size(IMf_dog,3)-ceil(1.5*tau):-1:2
    IMfr = IMf_dog(:,:,fr);
    IMpre = IMf_dog(:,:,fr-1);
    IMpost = IMf_dog(:,:,fr+1);
    selMax = IMfr == ordfilt2(IMfr, 9, ones(3));
    IMlocalMax = selMax & IMfr > IMpre & IMfr >= IMpost;
    skIm(IMlocalMax) = skIm(IMlocalMax) + IMfr(IMlocalMax).^2;
end
skIm = skIm ./ (300 + sum(~nans(:,:,2:end-ceil(1.5*tau)), 3));

step6_activity_raw = skIm;

% Post-process
skIm(~valid) = nan;
nanvals = isnan(skIm);
skIm(nanvals) = nanmedian(skIm(:));
mfSummary = nanmedfilt2(skIm, [5 5]);
skIm = skIm - mfSummary;
skIm(~valid) = nan;

step6_activity_final = skIm;

save(fullfile(outdir, 'matlab_step6_activity.mat'), ...
    'step6_activity_raw', 'step6_activity_final');

fprintf('\nDone. All intermediate results saved.\n');
