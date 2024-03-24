close all;
clear all;
%**********************************Read Image**************************
%select an image from the folder
folderPath = 'C:\Users\ricca\Desktop\Thesis\IMAGESTESTStdNames320';
% Set the output folder for results
outputFolder = 'C:\Users\ricca\Desktop\Thesis\dataTools\FiloDetect320';

% Micron to pixel conversion (replace with your actual value)
pixel = 9;

% Loop through all files in the folder
files = dir(fullfile(folderPath, '*.tif'));
for i = 1:245
    fileName = strcat(int2str(i), ".tif");
    fullPath = fullfile(folderPath, fileName);
    
    %disp('Please select an image');
    %[FileName,PathName] = uigetfile({'*.*', 'All Files (*.*)';'*.m;*.fig;*.mat;*.mdl','MATLAB Files (*.m,*.fig,*.mat,*.mdl)'},...
    %    'Choose a File','c:\users\rifaya\documents\matlab\work\');
    %II = imread([PathName FileName]);
    %Enter micron to pixel conversion for your image. 
    II = imread(fullPath);
    pixel='';
    if isempty(pixel)
        pixel=9;
    end
    
    %Following code use to output the result in any specified format
    loop=1;
    while (loop==0)
        disp('Please select [0-3] for visual output');
        disp('0 = No visual output')
        disp('1 = Output only final picture');
        disp('2 = Output all steps in one figure');
        disp('3 = Output all steps in seperate figures'); 
        mode=0;
            
        if isempty(mode)
            mode=0;
            loop=1;
        elseif (mode>=0 && mode<=3)
            loop=1;
        else
            disp('Not a valid option');
            mode=0;
            loop=0;
        end    
    end
    disp(sprintf('i: %s, Matrix type: %s, Size: %ix%i, Max: %f, Min: %f\n', fileName, class(II), size(II), max(II(:)), min(II(:))));
    [total_num, avg_length, length_array, result]=filopodia_detection(II,pixel,0);
    %disp(size(result))
    %imwrite(result, 'C:\Users\ricca\Desktop\test.png');
    outputFilename = fullfile(outputFolder, strrep(fileName, '.tif', '.tif'));
    imwrite(result, outputFilename);
end
%saveas(result,'C:\Users\admin\Desktop\test.png') ;

