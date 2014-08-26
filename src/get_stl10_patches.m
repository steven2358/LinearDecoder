function patches = get_stl10_patches()
% Returns patches and labels for STL10 data set. Checks if data is in
% '/data' folder and downloads from remote repository if necessary.
% Original URL:
% - http://ufldl.stanford.edu/wiki/resources/stl10_patches_100k.zip


url_data = 'http://ufldl.stanford.edu/wiki/resources/stl10_patches_100k.zip';
fname_data = 'stlSampledPatches.mat';

data_dir = '../data/';

%% load data

if (exist([data_dir fname_data],'file') ~= 2)
    t1 = tic;
    fprintf('Downloading data... ');
    unzip(url_data,data_dir);
    fprintf('%.2fs.\n',toc(t1));
end

t3 = tic;
fprintf('Loading data... ');
patches_struct = load([data_dir fname_data]);
patches = patches_struct.patches;
fprintf('%.2fs.\n',toc(t3));
