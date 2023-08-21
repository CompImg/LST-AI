function process_folder_LST_AI_LPA(mr_path, batches_path)
%
% requires BIDS conform database
%
% using CAT (version 12.8.1):
%
% 1. CAT Denoising (sanlm): all subfolders of the subject path are taken 
%    and CAT denoises the FLAIR images
% 2. LST segmentation (LPA): all subfolders of the subject path are taken 
%    and LST generates ples images based on the FLAIR images
% 3. LST extract values of interest: all subfolders of the subject path 
%    are taken and LST computes the lesion volumes and their number; 
%    this is done for the ples 
% 4. lesion mask: apply a threshold of 0.5 to the ples (because the same 
%    values is used to calculate total lesion cvolume) to generate a binary
%    lesion mask
% 5. move files: move files to derivatives folder in accordance with BIDS
%    file structure
%
% Inputs: 
% mr_path: This is the subject folder where the FLAIR images are stored
% batches_path: this is the folder where required matlab batches are stored
%
% !! you need to specify the name of the FLAIR files in line 61 !!

% Get a list of all the subfolders in 'path'
cd(mr_path)
% define session pattern
ses_pat = 'ses-' + wildcardPattern + '/';
% define subject pattern
sub_pat ='sub-' + wildcardPattern + '/';
% define derivatives path
deriv_path_tmp = strsplit(mr_path, '/');
deriv_path = strjoin([deriv_path_tmp(1:end-3), 'derivatives'], '/');
deriv_path_cat12 = strcat(deriv_path, '/', 'CAT-12.8.1');
deriv_path_LST_lpa = strcat(deriv_path, '/', 'LST-LPA-3.0.0');
% extract subject ID and session ID
mr_session_cell = replace(extract(mr_path, ses_pat), '/', '');
mr_session = mr_session_cell{1};
mr_subject_cell = replace(extract(mr_path, sub_pat), '/', '');
mr_subject = mr_subject_cell{1};

if isempty(mr_session) % Stop execution if the folder is empty
    return
end
if isempty(mr_subject) % Stop execution if the folder is empty
    return
end

log = mr_session_cell; 
[log{:,2}] = deal(mr_subject);
log = log(:,[2,1]);
[log{:,3:4}] = deal(false);
log = cell2table(log,...
    'VariableNames',{'mr_subject',...
    'mr_session',...
    'denoising_f2',...
    'lst_lpa'});

% define FLAIR file name
f2 = [mr_subject, '_', mr_session, '_space-mni_FLAIR.nii'];
% define output file names
sf2 = replace(f2, '_FLAIR', '_desc-sanlm_FLAIR');
msf2 = replace(sf2, '_desc-sanlm_', '_desc-sanlm#biascorr_');
lesion = replace(f2, '_FLAIR.', '_label-lesion_probseg.');
lesion_bin = replace(lesion, '_probseg.', '_mask.');
lst_report = replace(lesion, '_probseg.nii', '_probseg-report-LST-lpa.html');
lst_tlv = replace(f2, '_space-mni_FLAIR.nii', '_tlv.tsv');
% define paths to the files
% FLAIR 
path_f2 = fullfile(mr_path,f2);
path_sf2 = fullfile(mr_path, sf2);
path_sf2_deriv = fullfile(deriv_path_cat12, mr_subject, mr_session, 'anat', sf2);
path_msf2 = fullfile(mr_path, msf2);
path_msf2_deriv = fullfile(deriv_path_LST_lpa, mr_subject, mr_session, 'anat', msf2);
% lesion 
path_lesion = fullfile(mr_path, lesion);
path_lesion_deriv = fullfile(deriv_path_LST_lpa, mr_subject, mr_session, 'anat', lesion);
path_bin_lesion = fullfile(mr_path, lesion_bin);
path_bin_lesion_deriv = fullfile(deriv_path_LST_lpa, mr_subject, mr_session, 'anat', lesion_bin);
% report
path_lst_report = fullfile(mr_path, lst_report);
path_lst_report_deriv = fullfile(deriv_path_LST_lpa, mr_subject, mr_session, 'anat', lst_report);
% define LST_tlv.tsv path
lst_tlv_path = fullfile(mr_path, lst_tlv);
lst_tlv_path_deriv = fullfile(deriv_path_LST_lpa, mr_subject, mr_session, 'anat', lst_tlv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Denoising the t1 and the f2 using CAT12 sanlm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load CAT12 SANLM batch
load(fullfile(batches_path, 'cat12_sanlm_batch.mat'));
cat12_sanlm_batch = matlabbatch;

% Setting the options for CAT12 SANLM denoising
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.spm_type = 16;
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.prefix = 'sanlm_';
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.suffix = '';
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.intlim = 100;
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.rician = 0;
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.replaceNANandINF = 1;
cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.nlmfilter.optimized.NCstr = -Inf;

% Run CAT12 denoising
log.denoising_f2 = exist(fullfile(mr_path,sf2),'file')==2;
if ~log.denoising_f2 && exist(path_f2,'file') == 2
    try
        % display current case
        disp([datestr(datetime('now')),';', [mr_subject, '/', mr_session],': Denoising.'])
        % pass current FLAIR file path to batch
        cat12_sanlm_batch{1,1}.spm.tools.cat.tools.sanlm.data{1,1} = path_f2;
        spm_jobman('run',cat12_sanlm_batch);
        % rename output files
        movefile(strcat(mr_path, '/sanlm_', f2), ...
            path_sf2)
    catch
        disp([mr_subject, mr_session, ' ERROR: during denoising'])
    end
else
    disp([mr_subject, mr_session, ' : ', sf2, ' already exists!'])
end
log.denoising_f2 = exist(fullfile(mr_path,sf2),'file')==2;
% writetable(log,fullfile(mr_path,'log_pipeline.txt'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Running the LST segmentation (lpa)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load LST batch
load(fullfile(batches_path, 'lst_lpa_batch.mat'));
lst_lpa_batch = matlabbatch;

% Setting the options for LST
lst_lpa_batch{1,1}.spm.tools.LST.lpa.data_coreg{1, 1} = '';
lst_lpa_batch{1,1}.spm.tools.LST.lpa.html_report = 1; 

% Run LST LPA lesion segmentation
log.lst_lpa = exist(fullfile(mr_path,lesion),'file')==2;
if (~log.lst_lpa && log.denoising_f2)
    try
        % display current case
        disp([datestr(datetime('now')),';', [mr_subject, '/', mr_session],': Lesion segmentation.'])
        % pass current FLAIR file path to batch
        lst_lpa_batch{1,1}.spm.tools.LST.lpa.data_F2{1, 1} = path_sf2;
        spm_jobman('run',lst_lpa_batch);
        % rename output files
        % lesion map
        movefile(strcat(mr_path, "/ples_lpa_m", sf2), ...
            path_lesion)
        % other LST LPA files
        movefile(strcat(mr_path, "/m", mr_subject, '_', mr_session, '_space-mni_desc-sanlm_FLAIR.nii'),... 
                 strcat(path_msf2))
        movefile(strcat(mr_path, "/report_LST_lpa_m", mr_subject, '_', mr_session, '_space-mni_desc-sanlm_FLAIR.html'),... 
                 path_lst_report)
        % remove other folders and files
        rmdir(strcat('LST_lpa_m', mr_subject, '_', mr_session, '_space-mni_desc-sanlm_FLAIR'), 's')
        delete(strcat('LST_lpa_m', mr_subject, '_', mr_session, '_space-mni_desc-sanlm_FLAIR.mat'))
    catch
        disp([mr_subject, mr_session, ' ERROR: during lesion segmentation!'])
    end
else
    disp([mr_subject, mr_session, ' : ', lesion, ' already exists!'])
end
log.lst_lpa = exist(fullfile(mr_path,lesion),'file')==2;
% writetable(log,fullfile(mr_path,'log_pipeline.txt'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the volumes of interest using LST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load the batch for extraction of LST-values
load(fullfile(batches_path, 'lst_extract_values_batch.mat'))
lst_extract_batch = matlabbatch;

% delete the file if it already exists
if (exist(lst_tlv_path,'file'))
    delete(lst_tlv_path)
end

% generate new LST_tlv file
if (~exist(lst_tlv_path,'file'))
    delete LST*.csv
    try
        % display current case
        disp([datestr(datetime('now')),';', [mr_subject, '/', mr_session],': Extract VOI.'])
        % Setting the options for LST extract VOI
        lst_extract_batch{1,1}.spm.tools.LST.tlv.bin_thresh = 0.5;
        lst_extract_batch{1,1}.spm.tools.LST.tlv.data_lm{1,1} = path_lesion;
        spm_jobman('run',lst_extract_batch);
        % Rename output to LST_tlv.csv
        filename = dir(strcat(mr_path,'/LST_tlv*.csv'));
        tlv_data = readcell(strcat(filename.folder, '/', filename.name));
        writecell(tlv_data, lst_tlv_path, 'FileType','text', 'Delimiter','\t');
        delete 'LST_tlv*.csv'
    catch
        disp([mr_subject, '_', mr_session, ' ERROR: during LST data extraction!'])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create binary lesion mask from LST segmentation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (use threshold = 0.5, the same as for the tlv calculation)
% load the probability lesion map as nifti template
nifti_tmp = spm_vol(path_lesion);
% load the probability lesion map
lesion_bin_nifti = spm_vol(path_lesion);
lesion_bin_vol = spm_read_vols(lesion_bin_nifti);
lesion_bin_vol(isnan(lesion_bin_vol)) = 0;
% set a threshold to the probability lesion map to create a binary mask
les_mask = lesion_bin_vol >= 0.5;
% save the binary lesion mask
% important!!: define rescale slope as 1.0 so that voxel values are as
% defined and not modified by the rescale slope
nifti_tmp.pinfo(1)=1.0; %pinfo(1) defines the rescale slope
bin_les = nifti_tmp;
bin_les.fname = path_bin_lesion;
spm_write_vol(bin_les, les_mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% move files to derivatives folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp([datestr(datetime('now')),';', [mr_subject, '/', mr_session],': Extract VOI.'])
% denoised FLAIR (CAT12)
if (exist(path_sf2,'file'))
    % create derivatives directories if they do not already exist
    path_sf2_deriv_split = strsplit(path_sf2_deriv, '/');
    path_sf2_deriv_dir = strjoin(path_sf2_deriv_split(1:end-1), '/');
    if (~exist(path_sf2_deriv_dir, 'dir'))
        mkdir(path_sf2_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(path_sf2, ...
        path_sf2_deriv);
end

% denoised and bias corrected FLAIR (LST-LPA-3.0.0)
if (exist(path_msf2, 'file'))
    % create derivatives directories if they do not already exist
    path_msf2_deriv_split = strsplit(path_msf2_deriv, '/');
    path_msf2_deriv_dir = strjoin(path_msf2_deriv_split(1:end-1), '/');
    if (~exist(path_msf2_deriv_dir, 'dir'))
        mkdir(path_msf2_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(path_msf2, ...
        path_msf2_deriv);
end

% LST LPA report (LST-LPA-3.0.0)
if (exist(path_lst_report, 'file'))
    % create derivatives directories if they do not already exist
    path_lst_reportderiv_split = strsplit(path_lst_report_deriv, '/');
    path_lst_report_deriv_dir = strjoin(path_lst_reportderiv_split(1:end-1), '/');
    if (~exist(path_lst_report_deriv_dir, 'dir'))
        mkdir(path_lst_report_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(path_lst_report, ...
        path_lst_report_deriv);
end

% lesion probability map (LST-LPA-3.0.0)
if (exist(path_lesion, 'file'))
    % create derivatives directories if they do not already exist
    path_lesion_deriv_split = strsplit(path_lesion_deriv, '/');
    path_lesion_deriv_dir = strjoin(path_lesion_deriv_split(1:end-1), '/');
    if (~exist(path_lesion_deriv_dir, 'dir'))
        mkdir(path_lesion_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(path_lesion, ...
        path_lesion_deriv);
end

% binary lesion map (LST-LPA-3.0.0)
if (exist(path_bin_lesion, 'file'))
    % create derivatives directories if they do not already exist
    path_bin_lesion_deriv_split = strsplit(path_bin_lesion_deriv, '/');
    path_bin_lesion_deriv_dir = strjoin(path_bin_lesion_deriv_split(1:end-1), '/');
    if (~exist(path_bin_lesion_deriv_dir, 'dir'))
        mkdir(path_bin_lesion_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(path_bin_lesion, ...
        path_bin_lesion_deriv);
end

% LST lesion data (LST-LPA-3.0.0)
if (exist(lst_tlv_path, 'file'))
    % create derivatives directories if they do not already exist
    lst_tlv_path_deriv_split = strsplit(lst_tlv_path_deriv, '/');
    lst_tlv_path_deriv_dir = strjoin(lst_tlv_path_deriv_split(1:end-1), '/');
    if (~exist(lst_tlv_path_deriv_dir, 'dir'))
        mkdir(lst_tlv_path_deriv_dir)
    end
    % move the denoised FLAIR image to derivatives
    movefile(lst_tlv_path, ...
        lst_tlv_path_deriv);
end
disp([datestr(datetime('now')),';', [mr_subject, '/', mr_session],': DONE!'])

end

