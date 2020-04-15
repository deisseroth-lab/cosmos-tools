function varargout = analyzeVid_gui(varargin)
% ANALYZEVID_GUI MATLAB code for analyzeVid_gui.fig
%      USAGE:
%           Call analyzeVid_gui(vid)
%           Where vid is an [m x n x t] matrix containing a video to be
%           interactively analyzed. 
%
%      H = ANALYZEVID_GUI returns the handle to a new ANALYZEVID_GUI or the handle to
%      the existing singleton*.
%
%      ANALYZEVID_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ANALYZEVID_GUI.M with the given input arguments.
%
%      ANALYZEVID_GUI('Property','Value',...) creates a new ANALYZEVID_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before analyzeVid_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to analyzeVid_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help analyzeVid_gui

% Last Modified by GUIDE v2.5 23-Apr-2015 18:42:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @analyzeVid_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @analyzeVid_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before analyzeVid_gui is made visible.
function analyzeVid_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to analyzeVid_gui (see VARARGIN)

% Choose default command line output for analyzeVid_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% This sets up the initial plot - only do when we are invisible
% so window can get raised using analyzeVid_gui.
if strcmp(get(hObject,'Visible'),'off')
    plot(rand(5));
end

%%%%% Get the video from the input argument.
if nargin < 1
    error('Must provide a video as the first argument to analyzeVid_gui. Usage: analyzeVid_gui(vid) where vid is a variable containing a [m x n x t] image stack.');
end
axes(handles.axes1); cla;
handles.vid=varargin{1};
guidata(hObject, handles);
imagesc(handles.vid(:,:,1)'); axis square; axis off;
set(handles.maxVidText,'String', num2str(max(handles.vid(:)), '%10.2f'))

% UIWAIT makes analyzeVid_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = analyzeVid_gui_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('*.fig');
if ~isequal(file, 0)
    open(file);
end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
                     ['Close ' get(handles.figure1,'Name') '...'],...
                     'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.figure1)


% --- Executes on button press in alignToAtlasButton.
function alignToAtlasButton_Callback(hObject, eventdata, handles)
% hObject    handle to alignToAtlasButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in playButton.
function playButton_Callback(hObject, eventdata, handles)
% hObject    handle to playButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% try fps = evalin('base','fps'); catch; error('Specify fps variable in main workspace'); end

fps = 15;
handles.minVid = str2double(get(handles.minVidText,'String'));    
handles.maxVid = str2double(get(handles.maxVidText,'String'));    
scaleBar = [handles.minVid handles.maxVid];

handles.overlayAtlas = get(handles.overlayAtlasBox, 'Value');
handles.saveVid = get(handles.saveVidBox, 'Value');

guidata(hObject, handles);

if handles.saveVid
    fname = 'test';
else
   fname = [];
end

if handles.overlayAtlas
    try atlas = evalin('base','atlas'); catch; error('Specify "atlas" variable in main workspace, see'); end
    atlasOutline = GetAtlasOutline(atlas, false);
    OverlayAtlasVideo(handles.vid, atlasOutline, 15,scaleBar, fname);
else
    if ~isempty(fname)
        SaveVideo( handles.vid, fps, fname, scaleBar )
    else
        PlayVideo(handles.vid,fps,scaleBar);
    end
end



% --- Executes on button press in plotSelectionTracesButton.
function plotSelectionTracesButton_Callback(hObject, eventdata, handles)
% hObject    handle to plotSelectionTracesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
try atlas = evalin('base','atlas'); catch; error('Specify "atlas" variable in main workspace, see'); end
atlasOutline = GetAtlasOutline(atlas, false);
combined = OverlayAtlasOutline(max(handles.vid,[],3), atlasOutline, true, [0,0.1]);
[tt,x,y] = GetTraces(handles.vid, [0 0.1], combined, handles.axes1);
figure, plot(tt);

% --- Executes on button press in plotRegionalTracesButton.
function plotRegionalTracesButton_Callback(hObject, eventdata, handles)
% hObject    handle to plotRegionalTracesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

try atlas = evalin('base','atlas'); catch; error('Specify "atlas" variable in main workspace, see'); end
[traces, labels, trace_plot, label_plot, indiv_traces] = PlotRegionalTraces(handles.vid, atlas);


% --- Executes on button press in correlateTracesButton.
function correlateTracesButton_Callback(hObject, eventdata, handles)
% hObject    handle to correlateTracesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in saveVidBox.
function saveVidBox_Callback(hObject, eventdata, handles)
% hObject    handle to saveVidBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of saveVidBox



% --- Executes on button press in timeColorButton.
function timeColorButton_Callback(hObject, eventdata, handles)
% hObject    handle to timeColorButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[maxProj, Iarr, MnMx] = timeColorMapProj(handles.vid, 1, size(handles.vid,3));


% --- Executes on button press in clusterButton.
function clusterButton_Callback(hObject, eventdata, handles)
% hObject    handle to clusterButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in saveFigButton.
function saveFigButton_Callback(hObject, eventdata, handles)
% hObject    handle to saveFigButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

export_fig(handles.axes1, 'test_fig.pdf');
disp('saving fig');




% --- Executes on button press in overlayAtlasBox.
function overlayAtlasBox_Callback(hObject, eventdata, handles)
% hObject    handle to overlayAtlasBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of overlayAtlasBox



function minVidText_Callback(hObject, eventdata, handles)
% hObject    handle to minVidText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of minVidText as text
%        str2double(get(hObject,'String')) returns contents of minVidText as a double


% --- Executes during object creation, after setting all properties.
function minVidText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to minVidText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function maxVidText_Callback(hObject, eventdata, handles)
% hObject    handle to maxVidText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxVidText as text
%        str2double(get(hObject,'String')) returns contents of maxVidText as a double


% --- Executes during object creation, after setting all properties.
function maxVidText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxVidText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
