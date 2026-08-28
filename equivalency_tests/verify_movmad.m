%% Verify what MATLAB movmad actually computes
% Test: is it mean(|x - mean(x)|) or median(|x - median(x)|)?

rng(42);
x = randn(1, 1000);
k = 215;

% movmad output
m = movmad(x, k);

% Manual: at center point t=500
t = 500;
half_w = floor(k/2);
window = x(t-half_w:t+half_w);

% mean absolute deviation (from mean)
mad_mean = mean(abs(window - mean(window)));

% median absolute deviation (from median)
mad_median = median(abs(window - median(window)));

% MATLAB's mad() function
mad_matlab = mad(window, 0);  % flag=0: mean absolute deviation
mad_matlab_median = mad(window, 1);  % flag=1: median absolute deviation

fprintf('movmad output at t=%d: %.6f\n', t, m(t));
fprintf('mean(|x - mean(x)|): %.6f\n', mad_mean);
fprintf('median(|x - median(x)|): %.6f\n', mad_median);
fprintf('mad(window, 0) [mean]: %.6f\n', mad_matlab);
fprintf('mad(window, 1) [median]: %.6f\n', mad_matlab_median);

% Also check: does movmad match mad(x,1) or mad(x,0)?
fprintf('\nmovmad == mean(|x-mean(x)|)? diff = %.2e\n', abs(m(t) - mad_mean));
fprintf('movmad == median(|x-median(x)|)? diff = %.2e\n', abs(m(t) - mad_median));

% Now test with the actual data
data = load(fullfile(scan_dir(), 'registered_ds_act.mat'));
movie = double(data.movie_act);
alignHz = data.align_hz;
baselineWindow = ceil(4.0 * alignHz);
denoiseWindow = ceil(0.2 * alignHz);

% Prepare
IMf = movie;
nans = isnan(IMf);
valid = mean(nans, 3) < 0.33;
IMf(repmat(~valid, 1, 1, size(IMf,3))) = nan;
IMfden = smoothdata(IMf, 3, 'movmean', denoiseWindow, 'omitnan');
IMb = smoothdata(IMfden, 3, 'movmedian', baselineWindow, 'omitnan');
residual = IMfden - IMb;

% movmad on actual data
stdIM_raw = movmad(residual, baselineWindow, 3, 'omitmissing');

% Save raw movmad (before /0.6742 * denoiseWindow) for comparison
outdir = scan_dir();
raw_movmad_frame = stdIM_raw(:,:,1000);
save(fullfile(outdir, 'matlab_raw_movmad.mat'), 'raw_movmad_frame');

fprintf('\nRaw movmad at interior pixel (35,75), frame 1000: %.6f\n', stdIM_raw(35, 75, 1000));
fprintf('After scaling: %.6f\n', stdIM_raw(35, 75, 1000) / 0.6741891400433162 * denoiseWindow);
