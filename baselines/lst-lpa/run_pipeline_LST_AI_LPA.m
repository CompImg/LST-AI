function run_pipeline_LST_AI_LPA(mr_list_file)
% this function takes as input 
% mr_list_file: the path of a list (as .csv) of paths to the location of the FLAIR images (e.g., "anat" folder if BIDS conform)
% batches_path: the path of the folder which contains the required matlab batches
%
% A parallel processing pool is then opened in order to shorten the total processing time.
% In this parallel processing, the script "process_folder_LST_AI_LPA.m" is called. It includes all the necessary (pre-)processing steps 
% and details for the LST LPA lesion segmentation.
%
% optional settings to be defined in the script:
% default path of the list of paths to the location of FLAIR images: in line 18 -> mr_list_file = 'XX', XX defines the path of the list (.csv file)
% default path of the foder containing the required matlab batches: in line 19 -> batches = 'XX', XX defines the path of the folder 
% number of cores for the paralell processing: in line 28 -> parpool('local', XX, 'SpmdEnabled', false), XX defines the number of cores 
% path to spm software: in line 34 -> addpath XX, XX defines the path to the location of the spm software  

% Use the default location if the input is missing
if nargin == 0 
    mr_list_file = '/raid3/Tun/projects/LST_AI/LST-AI_scripts/LST-LPA/skull_bids_mr_list.csv';
    batches_path = '/raid3/Tun/projects/LST_AI/LST-AI_scripts/LST-LPA/matlab_batches';
end

mr_list = readtable(mr_list_file,'Delimiter',',');
mr_list = mr_list.filename;

% Open the parallel pool
warning('on')
try
    parpool('local',45,'SpmdEnabled',false)
catch
    warning('Parallelpool is already open. Consider closing the old one.')
end

% Add the SPM Toolbox (contains CAT version 12.8.1)
addpath /raid3/Tun/software/spm12_06_2022/spm12

% Run the image processing in parallel
parfor i = 1:length(mr_list)
    try
    	process_folder_LST_AI_LPA(mr_list{i}, batches_path)
    catch 
        warning(['Fehler bei der ID: ',mr_list{i}])
    end
end
end
