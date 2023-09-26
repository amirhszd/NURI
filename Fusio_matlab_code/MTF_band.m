function I_Filtered = MTF_band(I_MS_band,sensor,tag,ratio,band)

switch sensor
    case {'QB'}
        MTF_MS = [0.34 0.32 0.30 0.22]; % Band Order: B,G,R,NIR
    case {'IKONOS'}
        MTF_MS = [0.26,0.28,0.29,0.28]; % Band Order: B,G,R,NIR
    case 'All_03'
        MTF_MS = 0.3;
    case 'MS_029_PAN_015'
                MTF_MS = 0.29;
    case 'GeoEye1'
        MTF_MS = [0.23,0.23,0.23,0.23]; % Band Order: B,G,R,NIR
    case 'WV2'
        MTF_MS = [0.35 .* ones(1,7), 0.27];
    case {'WV3','WV3_4bands'}
        MTF_MS = [0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315];
        if strcmp(sensor,'WV3_4bands')
            tag = [2 3 5 7];
        end
    case {'HYP','HYP_14_33','HYP_16_31'}
        %VNIR
        MTF_MS(1:21)=0.27;
        MTF_MS(22:41)=0.28;
        MTF_MS(42:49)=0.26;
        MTF_MS(50:70)=0.26;
        %SWIR
        MTF_MS(71:100)=0.30;
        MTF_MS(101:130)=0.30;
        MTF_MS(131:177)=0.27;
        MTF_MS(177:242)=0.27;
        if strcmp(sensor,'HYP_14_33')
            tag = 14:33;
        elseif strcmp(sensor,'HYP_16_31')
            tag = 16:31;
        end
    case {'Ali_MS'}
        MTF_MS=[0.29,0.30,0.28,0.29,0.28,0.29,0.25,0.25,0.25];
    case 'none'
        MTF_MS = 0.29;
end

if (~isempty(tag) && isnumeric(tag))
    GNyq = MTF_MS(tag);
else
    GNyq = MTF_MS;
end


%%% MTF

N = 41;
fcut = 1/ratio;

alpha = sqrt((N*(fcut/2))^2/(-2*log(GNyq(band))));
H = fspecial('gaussian', N, alpha);
Hd = H./max(H(:));
h = fwind1(Hd,kaiser(N));
I_MS_LP = imfilter(I_MS_band,real(h),'replicate');

I_Filtered= double(I_MS_LP);

end