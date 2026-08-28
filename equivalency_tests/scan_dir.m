function dir_scan = scan_dir(scan)
%SCAN_DIR Directory of the reference scan used by the equivalency scripts.
%
%   Reads the session directory from the SUBCELL_TEST_DATA environment
%   variable and appends the scan folder name.
%
%   dir_scan = SCAN_DIR() uses the default reference scan.
%   dir_scan = SCAN_DIR(scan) uses the named scan folder.

if nargin < 1
    scan = 'test_scan_00001_20240924_110500';
end

dir_session = getenv('SUBCELL_TEST_DATA');
if isempty(dir_session)
    error(['SUBCELL_TEST_DATA is not set. Point it at the session directory ' ...
           'holding the scan folders, e.g. ''D:\iGluSnFR test data\750098\2024-09-24''.']);
end

dir_scan = fullfile(dir_session, scan);
if ~isfolder(dir_scan)
    error('Scan directory not found: %s', dir_scan);
end
end
