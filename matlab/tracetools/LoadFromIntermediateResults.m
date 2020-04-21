function neuron = LoadFromIntermediateResults(fname)
%%%% Loads the most recent results from the intermediate_results file saved
%%%% out by cnmf-e.

        DD = load(fname);
        fields = fieldnames(DD);
        
        tracefield = GetLastField(fields, 'temporal_');
        spatialfield = GetLastField(fields, 'spatial_');
        bgfield = GetLastField(fields, 'bg_');
        initfield = GetLastField(fields, 'initialization', false);
        
        spatialStruct = getfield(DD, spatialfield);
        bgStruct = getfield(DD, bgfield);
        initStruct = getfield(DD, initfield); 
        traceStruct = getfield(DD, tracefield); 
        
        neuron = Sources2D();
        neuron.A = spatialStruct.A;
        neuron.C = traceStruct.C;
        neuron.C_raw = traceStruct.C_raw;
        neuron.S = traceStruct.S;
        neuron.options = initStruct.neuron.options;
        neuron.P = traceStruct.P;
        neuron.b = bgStruct.b;
        neuron.f = bgStruct.f;
        neuron.W = bgStruct.W;
        neuron.b0 = bgStruct.b0;
        neuron.Fs = initStruct.neuron.Fs;
        neuron.frame_range = initStruct.neuron.frame_range;
        neuron.kernel = initStruct.neuron.kernel;
        neuron.file = initStruct.neuron.file;
        neuron.Cn = initStruct.Cn;
        neuron.ids = initStruct.neuron.ids;
        neuron.tags = initStruct.neuron.tags;
        neuron.PNF = initStruct.PNR;
        neuron.Cn = initStruct.Cn;
        
%         PNR = initStruct.PNR;
%         Cn = initStruct.Cn;
%         center = initStruct.center;